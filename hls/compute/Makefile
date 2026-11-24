# Plain-C++ C-sim for the compute core (no Vitis needed).
#   make csim               — synthetic-data testbench (tb/tb_dsc.cpp)
#   make vectors VEC="dir1 dir2 ..."
#                           — run the kernels on test-vector case folders, e.g. ones
#                             exported from the model (format: VECTORS.md)
#   make example-vectors    — generate a synthetic sparsity sweep with
#                             tools/dsc_vectors.py, check it, and run it
#   make clean
# Under Vitis HLS, use run_hls.tcl instead (it builds with ap_int types).

CXX      ?= g++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
INCLUDES  = -Iinclude -Igolden
PYTHON   ?= python3

KERNELS = src/requant.cpp src/depthwise.cpp src/pointwise.cpp src/pointwise_skip.cpp \
          golden/dsc_golden.cpp
HDRS    = include/*.h golden/*.h
BIN     = build/tb_dsc
VBIN    = build/tb_vectors
EXDIR   = build/vectors

.PHONY: csim vectors example-vectors clean

csim: $(BIN)
	./$(BIN)

$(BIN): $(KERNELS) tb/tb_dsc.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(KERNELS) tb/tb_dsc.cpp -o $@

$(VBIN): $(KERNELS) tb/tb_vectors.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(KERNELS) tb/tb_vectors.cpp -o $@

vectors: $(VBIN)
	@if [ -z "$(VEC)" ]; then echo 'usage: make vectors VEC="case_dir ..."'; exit 2; fi
	$(PYTHON) tools/dsc_vectors.py check $(VEC)
	./$(VBIN) $(VEC)

example-vectors: $(VBIN)
	@rm -rf $(EXDIR)
	$(PYTHON) tools/dsc_vectors.py example $(EXDIR)/dense_s1      --seed 1
	$(PYTHON) tools/dsc_vectors.py example $(EXDIR)/block30_s1    --seed 2 --block-prune 30
	$(PYTHON) tools/dsc_vectors.py example $(EXDIR)/block50_s1    --seed 3 --block-prune 50
	$(PYTHON) tools/dsc_vectors.py example $(EXDIR)/block70_s1    --seed 4 --block-prune 70
	$(PYTHON) tools/dsc_vectors.py example $(EXDIR)/block50_s2    --seed 5 --block-prune 50 --stride 2
	$(PYTHON) tools/dsc_vectors.py example $(EXDIR)/unstruct50_s1 --seed 6 --unstructured 50
	$(PYTHON) tools/dsc_vectors.py check $(EXDIR)/*
	./$(VBIN) $(EXDIR)/*

clean:
	rm -rf build
