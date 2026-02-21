#include "data_types.h"

void store_output(
    hls::stream<data_f> &s_out,
    data_f *output
)
{
#pragma HLS INLINE off

    for (int i = 0; i < 10; i++) {
#pragma HLS UNROLL
        output[i] = s_out.read();
    }

}
