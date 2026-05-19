#include "npu_ai_bench.hpp"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <latch>
#include <stdexcept>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

// bfloat16 stored as uint16_t; convert from float via IEEE-754 bit layout.
static uint16_t float_to_bf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = 0x7fff + ((bits >> 16) & 1);
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

static std::vector<uint32_t> load_instr_binary(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open instruction file: " + path);
    }
    auto byte_count = static_cast<std::size_t>(file.tellg());
    if (byte_count % sizeof(uint32_t) != 0) {
        throw std::runtime_error("Instruction file size not a multiple of 4: " + path);
    }
    file.seekg(0);
    std::vector<uint32_t> data(byte_count / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(data.data()), byte_count);
    return data;
}

NpuBenchResult run_npu_mul_bench(const NpuBenchConfig& config,
                                  std::latch& ready,
                                  std::shared_future<void> go,
                                  std::barrier<>& rep_barrier)
{
    const std::size_t elem_bytes = static_cast<std::size_t>(config.size) * sizeof(uint16_t);

    // ── Load instructions ────────────────────────────────────────────────────
    auto instr_v = load_instr_binary(config.insts_path);

    // ── XRT setup ────────────────────────────────────────────────────────────
    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(config.xclbin_path);
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());

    auto xkernels = xclbin.get_kernels();
    auto kit = std::find_if(xkernels.begin(), xkernels.end(),
        [](const xrt::xclbin::kernel& k) {
            return k.get_name().rfind("MLIR_AIE", 0) == 0;
        });
    if (kit == xkernels.end()) {
        throw std::runtime_error("MLIR_AIE kernel not found in " + config.xclbin_path);
    }
    auto kernel = xrt::kernel(context, kit->get_name());

    // ── Buffers ───────────────────────────────────────────────────────────────
    // Standard MLIR AIE 1-in / 1-out calling convention:
    //   group 1 – instruction stream (cacheable)
    //   group 3 – input  (host-only)
    //   group 4 – output (host-only)
    //   group 5,6,7 – dummy placeholders
    auto bo_instr    = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                               XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_in       = xrt::bo(device, elem_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_out      = xrt::bo(device, elem_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_tmp      = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    auto bo_ctrlpkts = xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
    auto bo_trace    = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

    std::memcpy(bo_instr.map<void*>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));

    uint16_t* buf_in = bo_in.map<uint16_t*>();
    std::srand(42);
    for (int i = 0; i < config.size; ++i) {
        float v = 4.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        buf_in[i] = float_to_bf16(v);
    }
    std::memset(bo_out.map<void*>(), 0, elem_bytes);

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    constexpr unsigned int opcode = 3;

    // ── Warmup (not timed, not part of the parallel overlap) ─────────────────
    for (int w = 0; w < config.warmup_iters; ++w) {
        kernel(opcode, bo_instr, static_cast<int>(instr_v.size()),
               bo_in, bo_out, bo_tmp, bo_ctrlpkts, bo_trace).wait();
    }

    // Signal that warmup is done and wait for the coordinated go signal.
    ready.count_down();
    go.wait();

    // ── Timed runs ───────────────────────────────────────────────────────────
    double total_us = 0.0;
    const auto run_start = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < config.repeats; ++rep) {
        rep_barrier.arrive_and_wait();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < config.iters; ++it)
            kernel(opcode, bo_instr, static_cast<int>(instr_v.size()),
                   bo_in, bo_out, bo_tmp, bo_ctrlpkts, bo_trace).wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    const auto run_end = std::chrono::high_resolution_clock::now();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // ── Metrics ──────────────────────────────────────────────────────────────
    // Each bfloat16 element does R aie::mac calls = R * 2 FLOPs.
    // AI = (size * r * 2) / (2 * size * sizeof(bf16)) = r / 2  FLOPs/Byte.
    const double average_ms      = total_us / config.repeats / 1000.0;
    const double seconds         = average_ms / 1000.0;
    const double flops           = static_cast<double>(config.size) * config.r * 2.0 * config.iters;
    const double bytes_moved     = 2.0 * elem_bytes * config.iters;
    const double bandwidth_gbps  = bytes_moved / seconds / 1e9;
    const double tflops          = flops / seconds / 1e12;
    const double arith_intensity = flops / bytes_moved;

    return {average_ms, tflops, bandwidth_gbps, arith_intensity, run_start, run_end};
}
