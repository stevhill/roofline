#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(cmd) \
    { \
        hipError_t e = cmd; \
        if (e != hipSuccess) { \
            printf("HIP error: %s:%d '%s'\n", __FILE__, __LINE__, hipGetErrorString(e)); \
            exit(EXIT_FAILURE); \
        } \
    }

__global__ void ai_kernel(half2* __restrict__ A,
                          half2* __restrict__ B,
                          int N,
                          int R)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop to cover large arrays
    for (int i = idx; i < N; i += stride) {

        // Load value once
        half2 a = A[i];
        half2 x = 0.0f;
        // Inline asm barrier to prevent compiler from optimizing away the computation
        asm volatile("" : "+v"(x));
        // Controlled compute: R FMAs
        half2 multiplier = __float2half2_rn(1.001f);
        #pragma unroll 8
        for (int r = 0; r < R; r++) {
            x = __hadd2(__hmul2(a, multiplier), x);
        }


        
        B[i] = x;
    }
}

int main(int argc, char** argv)
{
    int N = 1 << 20;   // ~256M elements (~1 GB total traffic)
    int R = 1;         // arithmetic intensity knob
    int repeats = 1;   // number of timed runs to average

    if (argc > 1) {
        R = std::atoi(argv[1]);
    }
    if (argc > 2) {
        repeats = std::atoi(argv[2]);
    }

    if (R < 1 || repeats < 1) {
        printf("Usage: %s [R>=1] [repeats>=1]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t bytes = N * sizeof(half2);

    half2* h_A = (half2*)malloc(bytes);
    half2* h_B = (half2*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
    }

    half2 *d_A, *d_B;
    CHECK(hipMalloc(&d_A, bytes));
    CHECK(hipMalloc(&d_B, bytes));

    CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Cap blocks to something reasonable (avoids oversubscription issues)
    numBlocks = min(numBlocks, 65535);

    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));

    CHECK(hipDeviceSynchronize());

    // Warm-up: run the kernel a few times to prime caches, JIT, and clocks
    int warmup_iters = 5;
    for (int wi = 0; wi < warmup_iters; ++wi) {
        ai_kernel<<<numBlocks, blockSize>>>(d_A, d_B, N, R);
    }
    CHECK(hipDeviceSynchronize());

    float total_ms = 0.0f;
    for (int rep = 0; rep < repeats; ++rep) {
        CHECK(hipEventRecord(start));

        ai_kernel<<<numBlocks, blockSize>>>(d_A, d_B, N, R);

        CHECK(hipEventRecord(stop));
        CHECK(hipEventSynchronize(stop));
        
        float ms = 0.0f;
        CHECK(hipEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    float ms = total_ms / repeats;

    CHECK(hipMemcpy(h_B, d_B, bytes, hipMemcpyDeviceToHost));

    // Compute metrics
    double seconds = ms / 1000.0;

    double flops = (double)N * R * 2;        // 1 FMA = 2 FLOPs
    double bytes_moved = (double)N * 2 * sizeof(half2);

    double gflops = flops / seconds / 1e9;
    double bandwidth = bytes_moved / seconds / 1e9;

    printf("R,");
    printf("Repeats,");
    printf("Time,");
    printf("GFLOPs,");
    printf("Bandwidth (GB/s),");
    printf("AI (FLOPs/Byte)");


    printf("\n");
    printf("%d,", R);
    printf("%d,", repeats);
    printf("%.3f,", ms);
    printf("%.2f,", gflops);
    printf("%.2f,", bandwidth);
    printf("%.2f", (double)flops / bytes_moved);
    printf("\n");

    hipFree(d_A);
    hipFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}