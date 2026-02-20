#include "load_input.h"
#include "conv1.h"
#include "pool1.h"
#include "conv2.h"
#include "pool2.h"
#include "dense3.h"
#include "store_output.h"

void CNN_MNIST_TOP(
    data_f *input,
    data_f *weight1,
    data_f *bias1,
    data_f *weight2,
    data_f *bias2,
    data_f *weight3,
    data_f *bias3,
    data_f *output
)
{

#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=784 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=weight1 offset=slave bundle=gmem1 depth=72 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=bias1 offset=slave bundle=gmem2 depth=8 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=weight2 offset=slave bundle=gmem3 depth=1152 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=bias2 offset=slave bundle=gmem4 depth=16 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=weight3 offset=slave bundle=gmem5 depth=4000 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=bias3 offset=slave bundle=gmem6 depth=10 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem7 depth=10

#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight1 bundle=control
#pragma HLS INTERFACE s_axilite port=bias1 bundle=control
#pragma HLS INTERFACE s_axilite port=weight2 bundle=control
#pragma HLS INTERFACE s_axilite port=bias2 bundle=control
#pragma HLS INTERFACE s_axilite port=weight3 bundle=control
#pragma HLS INTERFACE s_axilite port=bias3 bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_f input_buffer[28][28];
    data_f output_buffer[10];
    data_f W1[8][3][3];
    data_f W2[16][8][3][3];
    data_f W3[10][400];
    data_f B1[8];
    data_f B2[16];
    data_f B3[10];

//#pragma HLS ARRAY_PARTITION variable=input_buffer dim=1 type=cyclic factor=28
#pragma HLS ARRAY_PARTITION variable=output_buffer dim=0 type=complete

#pragma HLS BIND_STORAGE variable=input_buffer type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=output_buffer type=ram_2p impl=bram

    load_input(input, input_buffer, weight1, W1, weight2, W2, weight3, W3, bias1, B1, bias2, B2, bias3, B3);

#pragma HLS DATAFLOW

    hls::stream<vec8_f> s_conv1("s_conv1");
    hls::stream<vec8_f> s_pool1("s_pool1");
    hls::stream<vec16_f> s_conv2("s_conv2");
    hls::stream<vec16_f> s_pool2("s_pool2");

#pragma HLS STREAM variable=s_conv1 depth=128
#pragma HLS STREAM variable=s_pool1 depth=128
#pragma HLS STREAM variable=s_conv2 depth=128
#pragma HLS STREAM variable=s_pool2 depth=128

    conv1(input_buffer, W1, B1, s_conv1);
    pool1(s_conv1, s_pool1);
    conv2(s_pool1, W2, B2, s_conv2);
    pool2(s_conv2, s_pool2);
    dense3(s_pool2, W3, B3, output_buffer);

    // Outside Dataflow
    store_output(output_buffer, output);

}