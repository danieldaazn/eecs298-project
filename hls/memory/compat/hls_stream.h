// hls_stream.h — minimal stand-in for Xilinx's hls::stream, for plain-g++ C-sim ONLY.
//
// This directory is on the include path only in the Makefile build (no Xilinx
// install). Under Vitis HLS the real <hls_stream.h> is used; never add compat/ to
// run_hls.tcl.
//
// Semantics match Vitis C-sim (an unbounded FIFO; the depth pragma has no effect in
// C-sim), but misuse is stricter than Vitis: reading an empty stream, or destroying
// a stream that still holds data, aborts. Vitis only warns ("read while empty",
// "contains leftover data"), so these two checks catch producer/consumer count
// mismatches that Vitis C-sim would let slip through.

#ifndef HLS_STREAM_COMPAT_H
#define HLS_STREAM_COMPAT_H

#include <deque>
#include <stdio.h>
#include <stdlib.h>

namespace hls {

template <typename T>
class stream {
public:
    stream() : name_("(unnamed)") {}
    explicit stream(const char *name) : name_(name) {}
    ~stream() {
        if (!q_.empty()) {
            fprintf(stderr, "hls_stream compat: stream %s destroyed with %zu leftover elements\n",
                    name_, q_.size());
            abort();
        }
    }
    stream(const stream &) = delete;
    stream &operator=(const stream &) = delete;

    void write(const T &v) { q_.push_back(v); }
    T read() {
        if (q_.empty()) {
            fprintf(stderr, "hls_stream compat: read from empty stream %s\n", name_);
            abort();
        }
        T v = q_.front();
        q_.pop_front();
        return v;
    }
    void read(T &v) { v = read(); }
    bool empty() const { return q_.empty(); }
    bool full() const { return false; }
    size_t size() const { return q_.size(); }

private:
    std::deque<T> q_;
    const char *name_;
};

} // namespace hls

#endif // HLS_STREAM_COMPAT_H
