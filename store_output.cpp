#include "data_types.h"

void store_output(
    data_f output_buffer[10],
    data_f *output
)
{
#pragma HLS INLINE

    for (int i = 0; i < 10; i++) {
#pragma HLS UNROLL
        output[i] = output_buffer[i];
    }

}