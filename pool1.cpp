#include "data_types.h"

void pool1(
    hls::stream<vec8_f> &input,
    hls::stream<vec8_f> &output
)
{
#pragma HLS INLINE off

    data_f prev_row_buffer[8][26];
    data_f prev_pixel[8];

#pragma HLS ARRAY_PARTITION variable=prev_row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=prev_pixel complete

#pragma HLS DEPENDENCE variable=prev_row_buffer inter false
#pragma HLS DEPENDENCE variable=prev_pixel inter false

    for (int r = 0; r < 26; r++) {
        for (int c = 0; c < 26; c++) {
            vec8_f in_pixel = input.read();
            vec8_f out_pix;
#pragma HLS ARRAY_PARTITION variable=in_pixel.ch dim=0 type=complete
#pragma HLS ARRAY_PARTITION variable=out_pix.ch dim=0 type=complete
            for (int oc = 0; oc < 8; oc++) {
#pragma HLS PIPELINE II=1

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
                        data_f ans = (M1 > M2) ? M1 : M2;

                        out_pix.ch[oc] = ans;
                    }
                }
            }
            
            if((r&c&1) == 1) {
                output.write(out_pix);
            }
        }
    }
}
