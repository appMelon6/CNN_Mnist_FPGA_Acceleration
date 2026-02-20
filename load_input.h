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
);