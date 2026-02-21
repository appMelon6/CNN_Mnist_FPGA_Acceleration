#include "data_types.h"

void dense3(
    hls::stream<vec16_f> &input,
    hls::stream<data_f>  &weight_stream,
    hls::stream<data_f>  &bias_stream,
    hls::stream<data_f>  &output_stream
);
