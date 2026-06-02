#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <string>
#include <thread>

#include "gpu_ai_bench.hpp"
#include "npu_ai_bench.hpp"

namespace fs = std::filesystem;
using Clock  = std::chrono::high_resolution_clock;

struct MulBenchParams {
    int size_gpu      = 1 << 20;  // number of 16-bit elements for GPU (uses size/2 half2 pairs)
    int size_npu      = 1 << 20;  // number of 16-bit elements for NPU (matches existing artifacts)
    int r             = 2;
    int iters_gpu     = 1;        // kernel invocations per timed sample (scales GPU work/time)
    int iters_npu     = 1;        // kernel invocations per timed sample (scales NPU work/time)
    int repeats       = 3;        // timed repetitions averaged for the result
    int num_columns   = 8;
    int num_channels  = 2;
    int tile_size     = 8192;
    bool no_gpu       = false;
    bool no_npu       = false;
};

static fs::path find_repo_root()
{
    fs::path current = fs::current_path();
    while (!current.empty()) {
        if (fs::exists(current / "iron" / "operators" / "mul_bench" / "test.py"))
            return current;
        current = current.parent_path();
    }
    throw std::runtime_error("Unable to locate repository root");
}

static std::string artifact_stem(const MulBenchParams& p)
{
    return "MulBench_sz" + std::to_string(p.size_npu) +
           "_c"  + std::to_string(p.num_columns) +
           "_ch" + std::to_string(p.num_channels) +
           "_t"  + std::to_string(p.tile_size) +
           "_R"  + std::to_string(p.r) +
           "_npu2";
}

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s [--r R] [--iters-gpu N] [--iters-npu N] [--size-gpu N] [--size-npu N] [--repeats N] [--no-gpu] [--no-npu] [--json]\n"
        "  --r R          repeat count — controls arithmetic intensity (default: 2)\n"
        "  --iters-gpu N  kernel invocations per timed GPU sample (scales GPU work/time, default: 1)\n"
        "  --iters-npu N  kernel invocations per timed NPU sample (scales NPU work/time, default: 1)\n"
        "  --size-gpu N   GPU element count in 16-bit elements (default: 1048576)\n"
        "  --size-npu N   NPU element count in 16-bit elements (default: 1048576)\n"
        "  --repeats N    number of timed repetitions to average (default: 3)\n"
        "  --no-gpu       skip GPU benchmark\n"
        "  --no-npu       skip NPU benchmark\n"
        "  --json         print JSON result line to stdout\n",
        prog);
}

int main(int argc, char** argv)
{
    try {
        MulBenchParams params;
        bool json_output = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--r" && i + 1 < argc) {
                params.r = std::stoi(argv[++i]);
            } else if (arg == "--iters-gpu" && i + 1 < argc) {
                params.iters_gpu = std::stoi(argv[++i]);
            } else if (arg == "--iters-npu" && i + 1 < argc) {
                params.iters_npu = std::stoi(argv[++i]);
            } else if (arg == "--size-gpu" && i + 1 < argc) {
                params.size_gpu = std::stoi(argv[++i]);
            } else if (arg == "--size-npu" && i + 1 < argc) {
                params.size_npu = std::stoi(argv[++i]);
            } else if (arg == "--repeats" && i + 1 < argc) {
                params.repeats = std::stoi(argv[++i]);
            } else if (arg == "--no-gpu") {
                params.no_gpu = true;
            } else if (arg == "--no-npu") {
                params.no_npu = true;
            } else if (arg == "--json") {
                json_output = true;
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            } else {
                std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }

        if (params.no_gpu && params.no_npu) {
            throw std::runtime_error("--no-gpu and --no-npu both set: nothing to run");
        }

        const fs::path root = find_repo_root();

        // ── Shared timing parameters ─────────────────────────────────────────
        const int repeats      = params.repeats;
        const int warmup_iters = 3;

        GpuBenchConfig gpu_config;
        gpu_config.n            = params.size_gpu / 2;  // half2 pairs
        gpu_config.r            = params.r;
        gpu_config.iters        = params.iters_gpu;
        gpu_config.repeats      = repeats;
        gpu_config.warmup_iters = warmup_iters;

        NpuBenchConfig npu_config;
        if (!params.no_npu) {
            const fs::path xclbin_path = root / "build" / (artifact_stem(params) + ".xclbin");
            const fs::path insts_path  = root / "build" / (artifact_stem(params) + ".bin");
            if (!fs::exists(xclbin_path) || !fs::exists(insts_path)) {
                throw std::runtime_error(
                    "NPU artifacts not found — run: make ensure-artifacts\n"
                    "  missing: " + xclbin_path.string() + "\n"
                    "       or: " + insts_path.string());
            }
            npu_config.xclbin_path  = xclbin_path.string();
            npu_config.insts_path   = insts_path.string();
            npu_config.size         = params.size_npu;
            npu_config.r            = params.r;
            npu_config.iters        = params.iters_npu;
            npu_config.repeats      = repeats;
            npu_config.warmup_iters = warmup_iters;
        }

        // ── Synchronisation primitives ───────────────────────────────────────
        const int active = (params.no_gpu ? 0 : 1) + (params.no_npu ? 0 : 1);
        std::latch ready(active);
        std::promise<void> go_promise;
        std::shared_future<void> go = go_promise.get_future().share();
        // Barrier with arrival count = active devices; each repeat only starts
        // once every active device has finished the previous one.
        std::barrier<> rep_barrier(active);

        std::atomic<int> gpu_status{0};
        std::atomic<int> npu_status{0};
        GpuBenchResult gpu_result;
        NpuBenchResult npu_result;

        std::thread gpu_thread, npu_thread;

        if (!params.no_gpu) {
            gpu_thread = std::thread([&] {
                try {
                    gpu_result = run_gpu_ai_bench(gpu_config, ready, go, rep_barrier);
                } catch (const std::exception& e) {
                    std::cerr << "GPU error: " << e.what() << '\n';
                    gpu_status.store(1);
                    ready.count_down();
                }
            });
        }

        if (!params.no_npu) {
            npu_thread = std::thread([&] {
                try {
                    npu_result = run_npu_mul_bench(npu_config, ready, go, rep_barrier);
                } catch (const std::exception& e) {
                    std::cerr << "NPU error: " << e.what() << '\n';
                    npu_status.store(1);
                    ready.count_down();
                }
            });
        }

        ready.wait();

        auto wall_start = Clock::now();
        go_promise.set_value();

        if (gpu_thread.joinable()) gpu_thread.join();
        if (npu_thread.joinable()) npu_thread.join();
        auto wall_end = Clock::now();

        if (gpu_status.load() != 0 || npu_status.load() != 0) {
            std::cerr << "Benchmark failed\n";
            return EXIT_FAILURE;
        }

        const double wall_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(wall_end - wall_start)
                .count() / 1000.0;

        // ── Overlap analysis (only meaningful when both devices ran) ─────────
        const auto to_ms = [](auto dur) {
            return std::chrono::duration_cast<std::chrono::microseconds>(dur).count() / 1000.0;
        };

        double overlap_ms = 0.0;
        double overlap_pct_wall = 0.0;
        double overlap_pct_run  = 0.0;

        if (!params.no_gpu && !params.no_npu) {
            const double gpu_run_ms = to_ms(gpu_result.run_end - gpu_result.run_start);
            const double npu_run_ms = to_ms(npu_result.run_end - npu_result.run_start);
            const auto overlap_start = std::max(gpu_result.run_start, npu_result.run_start);
            const auto overlap_end   = std::min(gpu_result.run_end,   npu_result.run_end);
            overlap_ms = (overlap_end > overlap_start) ? to_ms(overlap_end - overlap_start) : 0.0;
            const double shorter_ms = std::min(gpu_run_ms, npu_run_ms);
            overlap_pct_wall = wall_ms > 0.0    ? overlap_ms / wall_ms    * 100.0 : 0.0;
            overlap_pct_run  = shorter_ms > 0.0 ? overlap_ms / shorter_ms * 100.0 : 0.0;
        }

        // ── Human-readable output (stderr when --json, stdout otherwise) ─────
        FILE* out = json_output ? stderr : stdout;

        if (!params.no_gpu) {
            std::fprintf(out, "\n[GPU]  (R=%d, size=%d, iters=%d, %d repeats)\n",
                         params.r, params.size_gpu, params.iters_gpu, repeats);
            std::fprintf(out, "  Time (ms):           %.3f\n",  gpu_result.average_ms);
            std::fprintf(out, "  TFLOPS:              %.4f\n",  gpu_result.tflops);
            std::fprintf(out, "  Bandwidth (GB/s):    %.2f\n",  gpu_result.bandwidth_gbps);
            std::fprintf(out, "  Arith intensity:     %.2f FLOPs/Byte\n", gpu_result.arith_intensity);
        }

        if (!params.no_npu) {
            std::fprintf(out, "\n[NPU]  (R=%d, size=%d, iters=%d, %d repeats)\n",
                         params.r, params.size_npu, params.iters_npu, repeats);
            std::fprintf(out, "  Time (ms):           %.3f\n",  npu_result.average_ms);
            std::fprintf(out, "  TFLOPS:              %.4f\n",  npu_result.tflops);
            std::fprintf(out, "  Bandwidth (GB/s):    %.2f\n",  npu_result.bandwidth_gbps);
            std::fprintf(out, "  Arith intensity:     %.2f FLOPs/Byte\n", npu_result.arith_intensity);
        }

        if (!params.no_gpu && !params.no_npu) {
            const double gpu_run_ms = to_ms(gpu_result.run_end - gpu_result.run_start);
            const double npu_run_ms = to_ms(npu_result.run_end - npu_result.run_start);
            std::fprintf(out, "\n[Overlap]\n");
            std::fprintf(out, "  GPU run window:      %.3f ms\n", gpu_run_ms);
            std::fprintf(out, "  NPU run window:      %.3f ms\n", npu_run_ms);
            if (overlap_ms > 0.0) {
                std::fprintf(out, "  Overlap duration:    %.3f ms\n",  overlap_ms);
                std::fprintf(out, "  %% of wall time:      %.1f%%\n",  overlap_pct_wall);
                std::fprintf(out, "  %% of shorter run:    %.1f%%\n",  overlap_pct_run);
            } else {
                std::fprintf(out, "  Overlap duration:    none (runs were sequential)\n");
            }
        }

        std::fprintf(out, "\n[Total wall time]  %.3f ms  (%d timed iterations)\n",
                    wall_ms, repeats);

        // ── Combined throughput (only when both devices ran) ──────────────────
        // Compute combined throughput per repeat (barrier-excluded), then
        // average those repeat-level TFLOPS values.
        double combined_tflops = 0.0;
        if (!params.no_gpu && !params.no_npu) {
            if (gpu_result.repeat_ms.size() != npu_result.repeat_ms.size()) {
                throw std::runtime_error("Mismatch in per-repeat timing data size between GPU and NPU");
            }
            const std::size_t rep_count = gpu_result.repeat_ms.size();
            if (rep_count == 0) {
                throw std::runtime_error("No per-repeat timing data available for combined throughput");
            }
            double combined_tflops_sum = 0.0;
            for (std::size_t i = 0; i < rep_count; ++i) {
                const double rep_window_ms = std::max(gpu_result.repeat_ms[i], npu_result.repeat_ms[i]);
                if (rep_window_ms <= 0.0) {
                    continue;
                }
                const double rep_tflops = ((gpu_result.flops + npu_result.flops) / 1e12) / (rep_window_ms / 1000.0);
                combined_tflops_sum += rep_tflops;
            }
            combined_tflops = combined_tflops_sum / static_cast<double>(rep_count);
            std::fprintf(out, "\n[Combined throughput]\n");
            std::fprintf(out, "  GPU:                 %.4f TFLOPS\n", gpu_result.tflops);
            std::fprintf(out, "  NPU:                 %.4f TFLOPS\n", npu_result.tflops);
            std::fprintf(out, "  Repeats averaged:    %zu\n", rep_count);
            std::fprintf(out, "  Combined:            %.4f TFLOPS\n", combined_tflops);
        }

        // ── JSON output ───────────────────────────────────────────────────────
        if (json_output) {
            std::printf("{\"r\":%d", params.r);
            if (!params.no_gpu) {
                std::printf(",\"size_gpu\":%d,\"iters_gpu\":%d"
                    ",\"gpu\":{\"average_ms\":%.6f,\"tflops\":%.8f,\"bandwidth_gbps\":%.4f,\"arith_intensity\":%.6f}",
                    params.size_gpu, params.iters_gpu,
                    gpu_result.average_ms, gpu_result.tflops, gpu_result.bandwidth_gbps, gpu_result.arith_intensity);
            }
            if (!params.no_npu) {
                std::printf(",\"size_npu\":%d,\"iters_npu\":%d"
                    ",\"npu\":{\"average_ms\":%.6f,\"tflops\":%.8f,\"bandwidth_gbps\":%.4f,\"arith_intensity\":%.6f}",
                    params.size_npu, params.iters_npu,
                    npu_result.average_ms, npu_result.tflops, npu_result.bandwidth_gbps, npu_result.arith_intensity);
            }
            if (!params.no_gpu && !params.no_npu) {
                std::printf(",\"combined_tflops\":%.8f", combined_tflops);
            }
            std::printf(",\"overlap_ms\":%.6f,\"wall_ms\":%.6f}\n", overlap_ms, wall_ms);
        }

        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
