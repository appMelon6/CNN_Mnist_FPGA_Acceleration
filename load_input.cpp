#include "data_types.h"

void load_input(
    data_f *input,
    data_f *weight1,
    data_f *bias1,
    data_f *weight2,
    data_f *bias2,
    data_f *weight3,
    data_f *bias3,

    hls::stream<data_f> &s_input,
    hls::stream<data_f> &s_w1,
    hls::stream<data_f> &s_b1,
    hls::stream<data_f> &s_w2,
    hls::stream<data_f> &s_b2,
    hls::stream<data_f> &s_w3,
    hls::stream<data_f> &s_b3
)
{
#pragma HLS INLINE off

    Stream_input:
    for (int i = 0; i < 784; i++) {
#pragma HLS PIPELINE II=1
        s_input.write(input[i]);
    }

    Stream_weight1:
    for (int i = 0; i < 72; i++) {
#pragma HLS PIPELINE II=1
        s_w1.write(weight1[i]);
    }

    Stream_bias1:
    for (int i = 0; i < 8; i++) {
#pragma HLS PIPELINE II=1
        s_b1.write(bias1[i]);
    }

    Stream_weight2:
    for (int i = 0; i < 1152; i++) {
#pragma HLS PIPELINE II=1
        s_w2.write(weight2[i]);
    }

    Stream_bias2:
    for (int i = 0; i < 16; i++) {
#pragma HLS PIPELINE II=1
        s_b2.write(bias2[i]);
    }

    Stream_weight3:
    for (int i = 0; i < 4000; i++) {
#pragma HLS PIPELINE II=1
        s_w3.write(weight3[i]);
    }

    Stream_bias3:
    for (int i = 0; i < 10; i++) {
#pragma HLS PIPELINE II=1
        s_b3.write(bias3[i]);
    }
}
