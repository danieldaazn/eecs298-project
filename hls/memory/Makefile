# Plain-C++ C-sim for the memory subsystem (no Vitis needed).
#   make csim   — build and run the testbench with native integer types and the
#                 compat/ hls::stream stand-in
#   make clean
# Under Vitis HLS, use run_hls.tcl instead (real hls_stream.h, ap_int types).

CXX      ?= g++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
INCLUDES  = -Iinclude -Icompat -I../compute/include

SRCS = src/mem_stages.cpp src/passthrough_top.cpp tb/tb_mem.cpp
BIN  = build/tb_mem

.PHONY: csim clean

csim: $(BIN)
	./$(BIN)

$(BIN): $(SRCS) include/*.h compat/*.h ../compute/include/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@

clean:
	rm -rf build
