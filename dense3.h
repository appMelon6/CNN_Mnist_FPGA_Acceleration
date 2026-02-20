#include "data_types.h"

void dense3(
    hls::stream<vec16_f> &input,
    data_f weight[10][400],
    data_f bias[10],
    data_f output[10]
);