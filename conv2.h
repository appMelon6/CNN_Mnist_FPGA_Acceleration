#include "data_types.h"

void conv2(
    hls::stream<vec8_f>  &input,
    hls::stream<data_f>  &weight_stream,
    hls::stream<data_f>  &bias_stream,
    hls::stream<vec16_f> &output
);
