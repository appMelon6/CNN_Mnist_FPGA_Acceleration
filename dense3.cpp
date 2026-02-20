#include "data_types.h"

void dense3(
    hls::stream<vec16_f> &input,
    data_f weight[10][400],
    data_f bias[10],
    data_f output[10]
)
{
#pragma HLS INLINE off

#pragma HLS ALLOCATION operation instances=mul limit=48

#pragma HLS ARRAY_PARTITION variable=bias dim=0 type=complete
#pragma HLS ARRAY_PARTITION variable=output dim=0 type=complete
#pragma HLS ARRAY_PARTITION variable=weight dim=2 type=cyclic factor=16

    acc_f acc[10];
#pragma HLS ARRAY_PARTITION variable=acc complete

    // Initialize accumulators with bias
    for (int o = 0; o < 10; o++) {
#pragma HLS UNROLL
        acc[o] = (acc_f)bias[o];
    }

    // 400/16 (parallelization)
    for (int i = 0; i < 25; i++) {
#pragma HLS PIPELINE II=10

        vec16_f pix = input.read();

#pragma HLS ARRAY_PARTITION variable=pix.ch complete

        // For each input channel
        for (int ic = 0; ic < 16; ic++) {
#pragma HLS UNROLL

            data_f val = pix.ch[ic];
            int base = i*16;

            // Update all outputs
            for (int o = 0; o < 10; o++) {
                acc[o] += (acc_f)val * weight[o][base + ic];
            }

        }
    }

    // Write final outputs
    for (int o = 0; o < 10; o++) {
#pragma HLS UNROLL

        acc_f temp = acc[o];

        output[o] = (data_f)temp;
    }
}
