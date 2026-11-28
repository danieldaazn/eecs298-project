# Build dsc_bringup natively on the KR260 (Kria Ubuntu: apt install build-essential).
#   make          build
#   make check    compile-only check, works on any Linux host (no board needed)
#   make hostsim  build and run it on any Linux host against the C-sim dsc_top instead
#                 of the hardware: checks everything but the register/DMA plumbing
#   make clean
# Shares the packers (hls/top/include/dsc_pack.h) and the golden model
# (hls/compute/golden) with the C-sim testbenches, so layouts can't drift.

ROOT     = ../..
CXX     ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label
INCLUDES = -I. -I$(ROOT)/hls/top/include -I$(ROOT)/hls/memory/include \
           -I$(ROOT)/hls/compute/include -I$(ROOT)/hls/compute/golden
SRCS     = dsc_bringup.cpp $(ROOT)/hls/compute/golden/dsc_golden.cpp

HLS_SRCS = $(ROOT)/hls/top/src/dsc_top.cpp $(ROOT)/hls/memory/src/mem_stages.cpp \
           $(ROOT)/hls/compute/src/requant.cpp $(ROOT)/hls/compute/src/depthwise.cpp \
           $(ROOT)/hls/compute/src/pointwise.cpp $(ROOT)/hls/compute/src/pointwise_skip.cpp

.PHONY: all check hostsim clean
all: dsc_bringup

dsc_bringup: $(SRCS) *.h $(ROOT)/hls/top/include/*.h $(ROOT)/hls/compute/golden/*.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $@

check:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fsyntax-only $(SRCS)

hostsim: dsc_bringup_hostsim
	./dsc_bringup_hostsim --reps 3 $(HOSTSIM_ARGS)

dsc_bringup_hostsim: $(SRCS) $(HLS_SRCS) *.h $(ROOT)/hls/top/include/*.h
	$(CXX) $(CXXFLAGS) -DDSC_HOST_SIM $(INCLUDES) -I$(ROOT)/hls/memory/compat $(SRCS) $(HLS_SRCS) -o $@

clean:
	rm -f dsc_bringup dsc_bringup_hostsim
