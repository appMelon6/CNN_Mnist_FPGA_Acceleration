#include "data_types.h"

void conv1(
    data_f input[28][28],
    data_f weight[8][3][3],
    data_f bias[8],
    hls::stream<vec8_f> &output
)
{
#pragma HLS INLINE off

    data_f window[3][3];
    data_f prev_row_buffer1[28];
    data_f prev_row_buffer2[28];

#pragma HLS ARRAY_PARTITION variable=window complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight dim=3 complete
#pragma HLS ARRAY_PARTITION variable=bias complete
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=2 dim=2

#pragma HLS DEPENDENCE variable=prev_row_buffer1 inter false
#pragma HLS DEPENDENCE variable=prev_row_buffer2 inter false
#pragma HLS DEPENDENCE variable=input inter false

    // Clear buffers
    for (int i = 0; i < 28; i++) {
#pragma HLS PIPELINE II=1
        prev_row_buffer1[i] = 0;
        prev_row_buffer2[i] = 0;
    }

    // Clear window
    for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
        for (int j = 0; j < 3; j++) {
#pragma HLS UNROLL
            window[i][j] = 0;
        }
    }

    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
#pragma HLS PIPELINE II=1

            bool valid = (r >= 2 && c >= 2);

            // Shift window left
            for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            data_f in_pixel = input[r][c];
            data_f row_minus_2 = prev_row_buffer1[c];
            data_f row_minus_1 = prev_row_buffer2[c];

            // Update window
            window[0][2] = row_minus_2;
            window[1][2] = row_minus_1;
            window[2][2] = in_pixel;

            // Update line buffers
            prev_row_buffer1[c] = row_minus_1;
            prev_row_buffer2[c] = in_pixel;

            if (valid) {
                vec8_f out_pix;
#pragma HLS ARRAY_PARTITION variable=out_pix.ch dim=0 type=complete
                for (int oc = 0; oc < 8; oc++) {

                    acc_f acc = (acc_f)bias[oc];

                    for (int kr = 0; kr < 3; kr++) {
#pragma HLS UNROLL
                        for (int kc = 0; kc < 3; kc++) {
#pragma HLS UNROLL
                            acc += weight[oc][kr][kc] *
                                   window[kr][kc];
                        }
                    }

                    if (acc < 0)
                        acc = 0;

                    out_pix.ch[oc] = (data_f)acc;
                }

                output.write(out_pix);
            }
        }
    }
}
