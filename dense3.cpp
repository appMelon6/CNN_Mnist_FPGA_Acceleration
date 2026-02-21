#include "data_types.h"

void dense3(
    hls::stream<vec16_f> &input,
    hls::stream<data_f>  &weight_stream,
    hls::stream<data_f>  &bias_stream,
    hls::stream<data_f>  &output_stream
)
{
#pragma HLS INLINE off

//#pragma HLS ALLOCATION operation instances=mul limit=48

    data_f weight[10][400];
    data_f bias[10];

#pragma HLS ARRAY_PARTITION variable=weight dim=2 cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=bias dim=0 type=complete

    Loading_weights_buffer:
    for (int o = 0; o < 10; o++) {
        for (int i = 0; i < 400; i++) {
#pragma HLS PIPELINE II=1
            weight[o][i] = weight_stream.read();
        }
    }

    Loading_bias_buffer:
    for (int o = 0; o < 10; o++) {
#pragma HLS PIPELINE II=1
        bias[o] = bias_stream.read();
    }

    acc_f acc[10];
#pragma HLS ARRAY_PARTITION variable=acc complete

    Initializing_accumulator:
    for (int o = 0; o < 10; o++) {
#pragma HLS UNROLL
        acc[o] = (acc_f)bias[o];
    }

    Dense_layer:
    for (int i = 0; i < 25; i++) {

        vec16_f pix = input.read();

#pragma HLS ARRAY_PARTITION variable=pix.ch complete

        Looping_over_Input_channels:
        for (int ic = 0; ic < 16; ic++) {
#pragma HLS PIPELINE II=1

            data_f val = pix.ch[ic];
            int base = i*16;

            Updating_outputs:
            for (int o = 0; o < 10; o++) {
#pragma HLS UNROLL
                acc[o] += (acc_f)val * weight[o][base + ic];
            }

        }
    }

    Writing_output:
    for (int o = 0; o < 10; o++) {
#pragma HLS PIPELINE II=1

        acc_f temp = acc[o];

        output_stream.write((data_f)temp);
    }
}
