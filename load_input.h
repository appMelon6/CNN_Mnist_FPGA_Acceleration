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
);
