// mem_stages.cpp — AXI read and line-buffer / sliding-window stages. See mem_stages.h.

#include "mem_stages.h"

void read_input(const axi_word_t *in, hls::stream<axi_word_t> &s_in) {
    // One sequential pass over the frame: Vitis infers a single long read burst
    // (split into max_read_burst_length chunks by the m_axi adapter).
READ_FRAME:
    for (int i = 0; i < IN_FRAME_BEATS; i++) {
#pragma HLS PIPELINE II=1
        s_in.write(in[i]);
    }
}

// Line buffer + window generator.
//
// Walks input pixels (y, x) in raster order over a (DW_H+1) x (DW_W+1) grid: the
// extra row/column are the bottom/right zero padding, fed as zeros without reading
// s_in. For each pixel it walks the DW_C channels, one per cycle (II=1):
//
//   col[0..2] = the 3 vertically stacked values of channel c at column x:
//               rows y-2 and y-1 from the line buffer, row y fresh from s_in.
//   line[0..1] (2 rows x DW_W x DW_C) is then shifted up for column x.
//   hist[c] holds columns x-2 and x-1 of channel c's window; the window is
//               [hist0 hist1 col] per row, and hist shifts left.
//
// The window completed at pixel (y, x) is centred on (y-1, x-1); it is emitted if
// that centre is a valid output position for the stride. Top/left padding comes from
// masking (rows y-2/y-1 < 0, columns x-2/x-1 < 0 read as zero), so neither buffer
// needs clearing between frames. (The bottom/right padding row/column also happen to
// flush zeros into the buffers, so after the first frame the masks are redundant;
// they are kept so the very first frame doesn't depend on power-up RAM contents.
// Mutation testing can't tell them apart in C-sim for that reason.)
void window_gen(hls::stream<axi_word_t> &s_in, hls::stream<win_t> &s_win, int stride) {
    // static: the RAMs keep their contents between calls in hardware, so C-sim must
    // too, or the back-to-back-frame test would not exercise stale data.
    static act_t line[DW_K - 1][DW_W][DW_C];
#pragma HLS ARRAY_PARTITION variable=line complete dim=1
    static act_t hist[DW_C][DW_K][DW_K - 1];
#pragma HLS ARRAY_PARTITION variable=hist complete dim=2
#pragma HLS ARRAY_PARTITION variable=hist complete dim=3
    axi_word_t cur = 0;

WG_ROWS:
    for (int y = 0; y <= DW_H; y++) {
    WG_COLS:
        for (int x = 0; x <= DW_W; x++) {
        WG_CH:
            for (int c = 0; c < DW_C; c++) {
#pragma HLS PIPELINE II=1
                // line[.][x][c] and hist[c] are revisited only DW_C (hist) or
                // (DW_W+1)*DW_C (line) iterations later, far beyond the pipeline
                // depth, so there is no true inter-iteration dependence. (DW_C >= 16 is
                // guaranteed by the DW_C % AXI_BYTES static_assert in mem_types.h; the
                // pipeline is a few stages deep. Revisit if DW_C ever gets smaller.)
#pragma HLS DEPENDENCE variable=line type=inter dependent=false
#pragma HLS DEPENDENCE variable=hist type=inter dependent=false
                const bool in_img = (y < DW_H) && (x < DW_W);
                const int lane = c % AXI_BYTES;
                if (in_img && lane == 0) cur = s_in.read();

                act_t col[DW_K];
                col[0] = (y >= 2 && x < DW_W) ? line[0][x][c] : (act_t)0;
                col[1] = (y >= 1 && x < DW_W) ? line[1][x][c] : (act_t)0;
                col[2] = in_img ? word_byte(cur, lane) : (act_t)0;
                if (x < DW_W) {
                    line[0][x][c] = col[1];
                    line[1][x][c] = col[2];
                }

                win_t win;
            WG_WIN_ROW:
                for (int r = 0; r < DW_K; r++) {
#pragma HLS UNROLL
                    const act_t h0 = (x == 0) ? (act_t)0 : hist[c][r][0];
                    const act_t h1 = (x == 0) ? (act_t)0 : hist[c][r][1];
                    win.v[r][0] = h0;
                    win.v[r][1] = h1;
                    win.v[r][2] = col[r];
                    hist[c][r][0] = h1;
                    hist[c][r][1] = col[r];
                }

                // Centre (y-1, x-1); stride 2 keeps even centres only.
                const bool centre_ok = (y >= 1) && (x >= 1) &&
                    (stride == 1 || (((y - 1) & 1) == 0 && ((x - 1) & 1) == 0));
                if (centre_ok) s_win.write(win);
            }
        }
    }
}
