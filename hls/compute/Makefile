# Plain-C++ C-sim for the compute core (no Vitis needed).
#   make csim   — build and run the testbench with native integer types
#   make clean
# Under Vitis HLS, use run_hls.tcl instead (it builds with ap_int types).

CXX      ?= g++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
INCLUDES  = -Iinclude -Igolden

SRCS = src/requant.cpp src/depthwise.cpp src/pointwise.cpp \
       golden/dsc_golden.cpp tb/tb_dsc.cpp
BIN  = build/tb_dsc

.PHONY: csim clean

csim: $(BIN)
	./$(BIN)

$(BIN): $(SRCS) include/*.h golden/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@

clean:
	rm -rf build
