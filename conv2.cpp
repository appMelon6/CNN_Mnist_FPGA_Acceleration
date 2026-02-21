#include "data_types.h"

void conv2(
    hls::stream<vec8_f>  &input,
    hls::stream<data_f>  &weight_stream,
    hls::stream<data_f>  &bias_stream,
    hls::stream<vec16_f> &output
)
{
#pragma HLS INLINE off
//#pragma HLS ALLOCATION instances=mul limit=108 operation

    data_f window[8][3][3];
    data_f prev_row_buffer1[8][13];
    data_f prev_row_buffer2[8][13];
    data_f weight[16][8][3][3];
    data_f bias[16];

#pragma HLS ARRAY_PARTITION variable=window dim=2 type=complete
#pragma HLS ARRAY_PARTITION variable=window dim=3 type=complete
#pragma HLS ARRAY_PARTITION variable=prev_row_buffer1 dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=prev_row_buffer2 dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=weight dim=3 type=complete
#pragma HLS ARRAY_PARTITION variable=weight dim=4 type=complete
#pragma HLS ARRAY_PARTITION variable=bias dim=0 type=complete

    Loading_weights_buffer:
    for (int oc = 0; oc < 16; oc++)
        for (int ic = 0; ic < 8; ic++)
            for (int kr = 0; kr < 3; kr++)
                for (int kc = 0; kc < 3; kc++) {
#pragma HLS PIPELINE II=1
                    weight[oc][ic][kr][kc] = weight_stream.read();
                }

    Loading_bias_buffer:
    for (int oc = 0; oc < 16; oc++) {
#pragma HLS PIPELINE II=1
        bias[oc] = bias_stream.read();
    }

    Clearing_previous_row_buffers:
    for (int ic = 0; ic < 8; ic++) {
        for (int i = 0; i < 13; i++) {
#pragma HLS PIPELINE II=1
            prev_row_buffer1[ic][i] = 0;
            prev_row_buffer2[ic][i] = 0;
        }
    }

    Main_convolution_plus_Relu:
    for (int r = 0; r < 13; r++) {
        for (int c = 0; c < 13; c++) {

            vec8_f in_pixel = input.read();
#pragma HLS ARRAY_PARTITION variable=in_pixel.ch complete

            Shift_window:
            for (int ic = 0; ic < 8; ic++) {
#pragma HLS UNROLL
                window[ic][0][0] = window[ic][0][1];
                window[ic][0][1] = window[ic][0][2];
                window[ic][1][0] = window[ic][1][1];
                window[ic][1][1] = window[ic][1][2];
                window[ic][2][0] = window[ic][2][1];
                window[ic][2][1] = window[ic][2][2];
            }

#pragma HLS DEPENDENCE variable=prev_row_buffer1 dependent=false type=inter
#pragma HLS DEPENDENCE variable=prev_row_buffer2 dependent=false type=inter
            
            Update_window_and_buffers:
            for (int ic = 0; ic < 8; ic++) {
#pragma HLS UNROLL
                window[ic][0][2] = prev_row_buffer2[ic][c];
                window[ic][1][2] = prev_row_buffer1[ic][c];
                window[ic][2][2] = in_pixel.ch[ic];

                prev_row_buffer2[ic][c] = prev_row_buffer1[ic][c];
                prev_row_buffer1[ic][c] = in_pixel.ch[ic];
            }

            if (r >= 2 && c >= 2) {

                vec16_f out_pix;
#pragma HLS ARRAY_PARTITION variable=out_pix.ch complete

                Convolution:
                for (int oc = 0; oc < 16; oc++) {
#pragma HLS PIPELINE II=1

                    acc_f acc = bias[oc];

                    for (int ic = 0; ic < 8; ic++) {
#pragma HLS UNROLL

                        for (int kr = 0; kr < 3; kr++) {
#pragma HLS UNROLL
                            for (int kc = 0; kc < 3; kc++) {
#pragma HLS UNROLL
                                acc += weight[oc][ic][kr][kc] *
                                       window[ic][kr][kc];
                            }
                        }
                    }

                    Relu:
                    if (acc < 0) {
                        acc = 0;
                    }

                    out_pix.ch[oc] = (data_f)acc;
                }

                Output_write:
                output.write(out_pix);
            }
        }
    }
}
