#pragma once

#include <barrier>
#include <chrono>
#include <cstdint>
#include <future>
#include <latch>

struct GpuBenchConfig {
    int n            = 1 << 20;  // half2 elements
    int r            = 2;
    int iters        = 1;        // kernel launches per timed sample
    int repeats      = 3;
    int warmup_iters = 3;
};

struct GpuBenchResult {
    double average_ms      = 0.0;
    double tflops          = 0.0;
    double bandwidth_gbps  = 0.0;
    double arith_intensity = 0.0;  // FLOPs / Byte
    std::chrono::high_resolution_clock::time_point run_start{};
    std::chrono::high_resolution_clock::time_point run_end{};
};

// Runs config.warmup_iters warmup iterations, then counts down `ready` to
// signal preparation is complete, then blocks on `go` until released.
// After `go` fires all config.repeats timed iterations execute back-to-back.
GpuBenchResult run_gpu_ai_bench(const GpuBenchConfig& config,
                                 std::latch& ready,
                                 std::shared_future<void> go,
                                 std::barrier<>& rep_barrier);
