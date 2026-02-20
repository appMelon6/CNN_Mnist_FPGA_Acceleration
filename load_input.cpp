#include "data_types.h"

void load_input(
    data_f *input,
    data_f input_buffer[28][28],
    data_f *weight1,
    data_f W1[8][3][3],
    data_f *weight2,
    data_f W2[16][8][3][3],
    data_f *weight3,
    data_f W3[10][400],
    data_f *bias1,
    data_f B1[8],
    data_f *bias2,
    data_f B2[16],
    data_f *bias3,
    data_f B3[10]
)
{
#pragma HLS INLINE off

    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
#pragma HLS PIPELINE
            input_buffer[r][c] = input[r*28 + c];
        }
    }

    // Conv1 weights
    for (int oc = 0; oc < 8; oc++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE
                W1[oc][i][j] = weight1[oc*9 + i*3 + j];
            }
        }
    }

    // Conv1 Biases
    for (int oc = 0; oc < 8; oc++) {
#pragma HLS PIPELINE
        B1[oc] = bias1[oc];
    }

    // Conv2 weights
    for (int ic = 0; ic < 8; ic++) {
        for (int oc = 0; oc < 16; oc++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE
                    W2[oc][ic][i][j] = weight2[oc*72 + ic*9 + i*3 + j];
                }
            }
        }
    }

    // Conv2 Biases
    for (int oc = 0; oc < 16; oc++) {
#pragma HLS PIPELINE
        B2[oc] = bias2[oc];
    }

    // Dense3 Weights
    for (int oc = 0; oc < 10; oc++) {
        for (int ic = 0; ic < 400; ic++) {
#pragma HLS PIPELINE
            W3[oc][ic] = weight3[oc*400 + ic];
        }
    }

    // Dense3 Biases
    for (int oc = 0; oc < 10; oc++) {
#pragma HLS PIPELINE
        B3[oc] = bias3[oc];
    }

}