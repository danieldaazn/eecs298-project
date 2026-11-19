# Plain-C++ C-sim for the fused top-level kernel (no Vitis needed).
#   make csim   — build and run the testbench with native integer types and the
#                 hls::stream stand-in from ../memory/compat
#   make clean
# Under Vitis HLS, use run_hls.tcl instead (real hls_stream.h, ap_int types).

CXX      ?= g++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
COMPUTE   = ../compute
MEMORY    = ../memory
INCLUDES  = -Iinclude -I$(MEMORY)/include -I$(MEMORY)/compat -I$(COMPUTE)/include -I$(COMPUTE)/golden

SRCS = src/dsc_top.cpp \
       $(MEMORY)/src/mem_stages.cpp \
       $(COMPUTE)/src/requant.cpp $(COMPUTE)/src/depthwise.cpp $(COMPUTE)/src/pointwise.cpp \
       $(COMPUTE)/src/pointwise_skip.cpp \
       $(COMPUTE)/golden/dsc_golden.cpp tb/tb_dsc_top.cpp
BIN  = build/tb_dsc_top

.PHONY: csim clean

csim: $(BIN)
	./$(BIN)

$(BIN): $(SRCS) include/*.h $(MEMORY)/include/*.h $(MEMORY)/compat/*.h $(COMPUTE)/include/*.h $(COMPUTE)/golden/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@

clean:
	rm -rf build
