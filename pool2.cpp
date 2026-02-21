#include "data_types.h"

void pool2(
    hls::stream<vec16_f> &input,
    hls::stream<vec16_f> &output
)
{
#pragma HLS INLINE off

    data_f prev_row_buffer[16][11];
    data_f prev_pixel[16];

#pragma HLS ARRAY_PARTITION variable=prev_row_buffer dim=1 type=complete
#pragma HLS ARRAY_PARTITION variable=prev_pixel dim=0 type=complete

    Max_pooling:
    for (int r = 0; r < 11; r++) {
        for (int c = 0; c < 11; c++) {
#pragma HLS PIPELINE II=1

            vec16_f in_pixel = input.read();
#pragma HLS ARRAY_PARTITION variable=in_pixel.ch complete

            vec16_f out_pix;
#pragma HLS ARRAY_PARTITION variable=out_pix.ch complete

            Looping_output_channels:
            for (int oc = 0; oc < 16; oc++) {
#pragma HLS UNROLL

                if ((r & 1) == 0) {
                    prev_row_buffer[oc][c] = in_pixel.ch[oc];
                }
                else {
                    if ((c & 1) == 0) {
                        prev_pixel[oc] = in_pixel.ch[oc];
                    }
                    else {
                        data_f L1 = prev_row_buffer[oc][c-1];
                        data_f R1 = prev_row_buffer[oc][c];
                        data_f L2 = prev_pixel[oc];
                        data_f R2 = in_pixel.ch[oc];

                        data_f M1 = (L1 > L2) ? L1 : L2;
                        data_f M2 = (R1 > R2) ? R1 : R2;

                        out_pix.ch[oc] = (M1 > M2) ? M1 : M2;
                    }
                }
            }

            if ((r & 1) && (c & 1)) {
                output.write(out_pix);
            }
        }
    }
}
