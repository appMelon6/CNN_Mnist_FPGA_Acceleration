#include "data_types.h"

void conv2(
    hls::stream<vec8_f> &input,
    data_f weight[16][8][3][3],
    data_f bias[16],
    hls::stream<vec16_f> &output
);