HIPCC   ?= hipcc
PYTHON  ?= conda run -n iron python

# ── build target ─────────────────────────────────────────────────────────────
TARGET  := mul_bench_parallel_test
SRC     := test.cpp gpu_ai_bench.cpp npu_ai_bench.cpp

CXXFLAGS  := -O3 -ffast-math -std=c++20
CXXFLAGS  += -I. -I../.. -I../../.. -I../../../include
CXXFLAGS  += -I/usr/include

LDFLAGS   := -lxrt++ -lxrt_coreutil

# ── NPU artifact parameters (must match MulBenchParams defaults in test.cpp) ──
NPU_SIZE      ?= 16777216   # 1 << 24
NPU_R         ?= 2
NPU_COLUMNS   ?= 8
NPU_CHANNELS  ?= 2
NPU_TILE_SIZE ?= 8192

ARTIFACT_STEM := MulBench_sz$(NPU_SIZE)_c$(NPU_COLUMNS)_ch$(NPU_CHANNELS)_t$(NPU_TILE_SIZE)_R$(NPU_R)_npu2
REPO_ROOT     := ../../..
XCLBIN        := $(REPO_ROOT)/build/$(ARTIFACT_STEM).xclbin
INSTBIN       := $(REPO_ROOT)/build/$(ARTIFACT_STEM).bin

NPU_PY_ARGS   := --size $(NPU_SIZE) --r $(NPU_R) \
                 --num-columns $(NPU_COLUMNS) \
                 --num-channels $(NPU_CHANNELS) \
                 --tile-size $(NPU_TILE_SIZE)

.PHONY: all clean run ensure-artifacts

all: $(TARGET)

$(TARGET): $(SRC) gpu_ai_bench.hpp npu_ai_bench.hpp
	$(HIPCC) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $@

# Compile NPU xclbin/bin artifacts via the IRON Python toolchain (one-time).
$(XCLBIN) $(INSTBIN):
	cd $(REPO_ROOT) && $(PYTHON) iron/operators/mul_bench/run_mul_bench_npu.py $(NPU_PY_ARGS)

ensure-artifacts: $(XCLBIN) $(INSTBIN)

# Build binary + ensure NPU artifacts exist, then run.
run: $(TARGET) ensure-artifacts
	./$(TARGET)

clean:
	rm -f $(TARGET)
