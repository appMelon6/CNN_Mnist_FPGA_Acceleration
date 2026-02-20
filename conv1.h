#include "data_types.h"

void conv1(
    data_f input[28][28],
    data_f weight[8][3][3],
    data_f bias[8],
    hls::stream<vec8_f> &output
);