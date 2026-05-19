#include "gpu_ai_bench.hpp"

#include <algorithm>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <latch>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                                 \
    do {                                                                           \
        hipError_t error = cmd;                                                    \
        if (error != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error: %s:%d '%s'\n", __FILE__, __LINE__,   \
                         hipGetErrorString(error));                                \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    } while (false)

__global__ void ai_kernel(half2* __restrict__ a,
                          half2* __restrict__ b,
                          int n,
                          int r)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        half2 value = a[i];
        half2 multiplier = __float2half2_rn(1.001f);
        // Eight independent accumulators give the wavefront scheduler enough
        // independent FMAs to fully hide the ~5-cycle v_pk_fma_f16 latency.
        // Inline asm forces v_pk_fma_f16 — -ffast-math would otherwise
        // collapse acc += val*mult (R times) into a pure multiply chain.
        half2 acc0 = __float2half2_rn(0.0f);
        half2 acc1 = __float2half2_rn(0.0f);
        half2 acc2 = __float2half2_rn(0.0f);
        half2 acc3 = __float2half2_rn(0.0f);
        half2 acc4 = __float2half2_rn(0.0f);
        half2 acc5 = __float2half2_rn(0.0f);
        half2 acc6 = __float2half2_rn(0.0f);
        half2 acc7 = __float2half2_rn(0.0f);
        int rep = 0;
        for (; rep <= r - 8; rep += 8) {
            asm volatile(
                "v_pk_fma_f16 %0, %8, %9, %0\n\t"
                "v_pk_fma_f16 %1, %8, %9, %1\n\t"
                "v_pk_fma_f16 %2, %8, %9, %2\n\t"
                "v_pk_fma_f16 %3, %8, %9, %3\n\t"
                "v_pk_fma_f16 %4, %8, %9, %4\n\t"
                "v_pk_fma_f16 %5, %8, %9, %5\n\t"
                "v_pk_fma_f16 %6, %8, %9, %6\n\t"
                "v_pk_fma_f16 %7, %8, %9, %7"
                : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3),
                  "+v"(acc4), "+v"(acc5), "+v"(acc6), "+v"(acc7)
                : "v"(value), "v"(multiplier));
        }
        for (; rep < r; ++rep) {
            asm volatile("v_pk_fma_f16 %0, %1, %2, %0"
                         : "+v"(acc0) : "v"(value), "v"(multiplier));
        }
        b[i] = __hadd2(__hadd2(__hadd2(acc0, acc1), __hadd2(acc2, acc3)),
                       __hadd2(__hadd2(acc4, acc5), __hadd2(acc6, acc7)));
    }
}

GpuBenchResult run_gpu_ai_bench(const GpuBenchConfig& config,
                                 std::latch& ready,
                                 std::shared_future<void> go,
                                 std::barrier<>& rep_barrier)
{
    const int n            = config.n;
    const int r            = config.r;
    const int iters        = config.iters;
    const int repeats      = config.repeats;
    const int warmup_iters = config.warmup_iters;

    size_t bytes = static_cast<size_t>(n) * sizeof(half2);
    half2* host_a = static_cast<half2*>(std::malloc(bytes));
    half2* host_b = static_cast<half2*>(std::malloc(bytes));
    if (!host_a || !host_b) {
        std::fprintf(stderr, "GPU: failed to allocate host buffers\n");
        std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) host_a[i] = 1.0f;

    half2* device_a = nullptr;
    half2* device_b = nullptr;
    CHECK(hipMalloc(&device_a, bytes));
    CHECK(hipMalloc(&device_b, bytes));
    CHECK(hipMemcpy(device_a, host_a, bytes, hipMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = std::min((n + block_size - 1) / block_size, 65535);

    hipEvent_t ev_start = nullptr;
    hipEvent_t ev_stop  = nullptr;
    CHECK(hipEventCreate(&ev_start));
    CHECK(hipEventCreate(&ev_stop));

    // ── Warmup (not timed, not part of the parallel overlap) ─────────────────
    CHECK(hipDeviceSynchronize());
    for (int w = 0; w < warmup_iters; ++w) {
        ai_kernel<<<num_blocks, block_size>>>(device_a, device_b, n, r);
    }
    CHECK(hipDeviceSynchronize());

    // Signal that warmup is done and wait for the coordinated go signal.
    ready.count_down();
    go.wait();

    // ── Timed runs ───────────────────────────────────────────────────────────
    float total_ms = 0.0f;
    const auto run_start = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < repeats; ++rep) {
        rep_barrier.arrive_and_wait();
        CHECK(hipEventRecord(ev_start));
        for (int it = 0; it < iters; ++it)
            ai_kernel<<<num_blocks, block_size>>>(device_a, device_b, n, r);
        CHECK(hipEventRecord(ev_stop));
        CHECK(hipEventSynchronize(ev_stop));

        float elapsed_ms = 0.0f;
        CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
        total_ms += elapsed_ms;
    }
    const auto run_end = std::chrono::high_resolution_clock::now();

    CHECK(hipMemcpy(host_b, device_b, bytes, hipMemcpyDeviceToHost));

    // ── Metrics ──────────────────────────────────────────────────────────────
    // Count individual fp16 elements (n half2 = 2n fp16); each does R MACs
    // (multiply-accumulate = 2 FLOPs), matching the NPU per-element convention.
    // AI = (2n * r * 2) / (n * 2 * sizeof(half2)) = r / 2  FLOPs/Byte.
    const double average_ms      = total_ms / repeats;
    const double seconds         = average_ms / 1000.0;
    const double flops           = static_cast<double>(n) * 2.0 * r * 2.0 * iters;
    const double bytes_moved     = static_cast<double>(n) * 2.0 * sizeof(half2) * iters;

    GpuBenchResult result;
    result.average_ms      = average_ms;
    result.tflops          = flops / seconds / 1e12;
    result.bandwidth_gbps  = bytes_moved / seconds / 1e9;
    result.arith_intensity = flops / bytes_moved;
    result.run_start       = run_start;
    result.run_end         = run_end;

    CHECK(hipEventDestroy(ev_start));
    CHECK(hipEventDestroy(ev_stop));
    CHECK(hipFree(device_a));
    CHECK(hipFree(device_b));
    std::free(host_a);
    std::free(host_b);

    return result;
}
