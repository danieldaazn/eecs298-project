# libdsc.so for server.py. Builds on the KR260 and on any Linux PC (SIM mode there).
ROOT     = ../..
CXX     ?= g++
CXXFLAGS ?= -std=c++17 -O3 -fPIC -Wall -Wextra -Wno-unknown-pragmas -ffp-contract=off
INCLUDES = -I$(ROOT)/board/sw -I$(ROOT)/hls/top/include -I$(ROOT)/hls/memory/include \
           -I$(ROOT)/hls/compute/include -I$(ROOT)/hls/compute/golden
SRCS     = libdsc.cpp $(ROOT)/hls/compute/golden/dsc_golden.cpp

libdsc.so: $(SRCS) $(ROOT)/board/sw/*.h $(ROOT)/hls/top/include/*.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared $(SRCS) -o $@

.PHONY: clean
clean:
	rm -f libdsc.so
