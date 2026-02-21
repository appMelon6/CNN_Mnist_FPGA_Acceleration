#include "data_types.h"

void conv1(
    hls::stream<data_f> &s_input,
    hls::stream<data_f> &s_weight,
    hls::stream<data_f> &s_bias,
    hls::stream<vec8_f> &output
);
