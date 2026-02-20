#include "data_types.h"

void conv2(
    hls::stream<vec8_f> &input,
    data_f weight[16][8][3][3],
    data_f bias[16],
    hls::stream<vec16_f> &output
)
{
#pragma HLS INLINE off

    data_f window[8][3][3];
    data_f prev_row_buffer1[8][13];
    data_f prev_row_buffer2[8][13];

#pragma HLS ARRAY_PARTITION variable=window dim=2 complete
#pragma HLS ARRAY_PARTITION variable=window dim=3 complete
#pragma HLS ARRAY_PARTITION variable=prev_row_buffer1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=prev_row_buffer2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=3 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=4 complete

    // Previous Row buffers
    for (int ic = 0; ic < 8; ic++) {
        for (int i = 0; i < 13; i++) {
            prev_row_buffer1[ic][i] = 0;
            prev_row_buffer2[ic][i] = 0;
        }
    }

    // Clear window
    for (int ic = 0; ic < 8; ic++)
        for (int i = 0; i < 3; i++)
#pragma HLS UNROLL
            for (int j = 0; j < 3; j++)
#pragma HLS UNROLL
                window[ic][i][j] = 0;

    for (int r = 0; r < 13; r++) {
        for (int c = 0; c < 13; c++) {

            bool valid = (r >= 2 && c >= 2);

            vec8_f in_pixel = input.read();
#pragma HLS ARRAY_PARTITION variable=in_pixel.ch complete

            for (int ic = 0; ic < 8; ic++)
#pragma HLS UNROLL

            // Shift window
            for (int ic = 0; ic < 8; ic++) {
#pragma HLS UNROLL
                for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
                    window[ic][i][0] = window[ic][i][1];
                    window[ic][i][1] = window[ic][i][2];
                }
            }
            // Update window and buffers
            for (int ic = 0; ic < 8; ic++) {
#pragma HLS UNROLL
                data_f row2 = prev_row_buffer1[ic][c];
                data_f row1 = prev_row_buffer2[ic][c];

                window[ic][0][2] = row2;
                window[ic][1][2] = row1;
                window[ic][2][2] = in_pixel.ch[ic];

                prev_row_buffer1[ic][c] = row1;
                prev_row_buffer2[ic][c] = in_pixel.ch[ic];
            }

            if (valid) {
                vec16_f out_pix;
#pragma HLS ARRAY_PARTITION variable=out_pix.ch dim=0 type=complete
                for (int oc = 0; oc < 16; oc++) {
#pragma HLS PIPELINE II=1

                    // Actual convolution
                    acc_f acc = bias[oc];

                    for (int ic = 0; ic < 8; ic++) {
                        for (int kr = 0; kr < 3; kr++) {
#pragma HLS UNROLL
                            for (int kc = 0; kc < 3; kc++) {
#pragma HLS UNROLL
                                acc += weight[oc][ic][kr][kc] *
                                       window[ic][kr][kc];
                            }
                        }
                    }

                    if (acc < 0) {
                        acc = 0;
                    }

                    out_pix.ch[oc] = (data_f)acc;
                }

                output.write(out_pix);
            }
        }
    }
}
