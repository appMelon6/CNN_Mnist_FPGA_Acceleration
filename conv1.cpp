#include "data_types.h"

void conv1(
    hls::stream<data_f> &s_input,
    hls::stream<data_f> &s_weight,
    hls::stream<data_f> &s_bias,
    hls::stream<vec8_f> &output
)
{
#pragma HLS INLINE off

    data_f weight[8][3][3];
    data_f bias[8];
    data_f window[3][3];
    data_f prev_row_buffer1[28];
    data_f prev_row_buffer2[28];

#pragma HLS ARRAY_PARTITION variable=window dim=0 type=complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 type=complete
#pragma HLS ARRAY_PARTITION variable=weight dim=3 type=complete
#pragma HLS ARRAY_PARTITION variable=bias dim=0 type=complete

    Loading_weights_buffer:
    for (int i = 0; i < 8; i++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
#pragma HLS PIPELINE II=1
                weight[i][r][c] = s_weight.read();
            }
        }
    }

    Loading_bias_buffer:
    for (int i = 0; i < 8; i++) {
#pragma HLS PIPELINE II=1
        bias[i] = s_bias.read();
    }

    Clearing_previous_row_buffers:
    for (int i = 0; i < 28; i++) {
#pragma HLS PIPELINE II=1
        prev_row_buffer1[i] = 0;
        prev_row_buffer2[i] = 0;
    }

    Clearing_sliding_window:
    for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
        for (int j = 0; j < 3; j++) {
#pragma HLS UNROLL
            window[i][j] = 0;
        }
    }

    Main_convolution_plus_Relu:
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
#pragma HLS PIPELINE II=1

            data_f in_pixel = s_input.read();

            bool valid = (r >= 2 && c >= 2);

            Shift_window:
            for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            Update_window_and_buffers:
            window[0][2] = prev_row_buffer2[c];
            window[1][2] = prev_row_buffer1[c];
            window[2][2] = in_pixel;

            prev_row_buffer2[c] = prev_row_buffer1[c];
            prev_row_buffer1[c] = in_pixel;

            if (valid) {
                vec8_f out_pix;
#pragma HLS ARRAY_PARTITION variable=out_pix.ch complete dim=0

                Convolution:
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

                    Relu:
                    if (acc < 0)
                        acc = 0;

                    out_pix.ch[oc] = (data_f)acc;
                }

                Output_write:
                output.write(out_pix);
            }
        }
    }
}
