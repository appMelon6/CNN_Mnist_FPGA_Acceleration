#include "load_input.h"
#include "conv1.h"
#include "pool1.h"
#include "conv2.h"
#include "pool2.h"
#include "dense3.h"
#include "store_output.h"

void CNN_MNIST_TOP(
    data_f *input,      //1x28x28
    data_f *weight1,    //8x3x3
    data_f *bias1,      //8
    data_f *weight2,    //16x8x3x3
    data_f *bias2,      //16
    data_f *weight3,    //10x400
    data_f *bias3,      //10
    data_f *output      //10
)
{

#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=784 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=weight1 offset=slave bundle=gmem1 depth=72 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=bias1 offset=slave bundle=gmem1 depth=8 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=weight2 offset=slave bundle=gmem2 depth=1152 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=bias2 offset=slave bundle=gmem2 depth=16 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=weight3 offset=slave bundle=gmem3 depth=4000 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=bias3 offset=slave bundle=gmem3 depth=10 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem4 depth=10 max_write_burst_length=16

#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weight1 bundle=control
#pragma HLS INTERFACE s_axilite port=bias1 bundle=control
#pragma HLS INTERFACE s_axilite port=weight2 bundle=control
#pragma HLS INTERFACE s_axilite port=bias2 bundle=control
#pragma HLS INTERFACE s_axilite port=weight3 bundle=control
#pragma HLS INTERFACE s_axilite port=bias3 bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    hls::stream<data_f> s_input;
    hls::stream<data_f> s_w1, s_b1;
    hls::stream<data_f> s_w2, s_b2;
    hls::stream<data_f> s_w3, s_b3;

#pragma HLS STREAM variable=s_input depth=1024
#pragma HLS STREAM variable=s_w1 depth=128
#pragma HLS STREAM variable=s_w2 depth=2048
#pragma HLS STREAM variable=s_w3 depth=4096

    hls::stream<vec8_f>  s_conv1("s_conv1");
    hls::stream<vec8_f>  s_pool1("s_pool1");
    hls::stream<vec16_f> s_conv2("s_conv2");
    hls::stream<vec16_f> s_pool2("s_pool2");
    hls::stream<data_f> s_out("s_out");

#pragma HLS STREAM variable=s_conv1 depth=128
#pragma HLS STREAM variable=s_pool1 depth=128
#pragma HLS STREAM variable=s_conv2 depth=128
#pragma HLS STREAM variable=s_pool2 depth=128
#pragma HLS STREAM variable=s_out depth=128

#pragma HLS DATAFLOW

    load_input( input, weight1, bias1, weight2, bias2, weight3, bias3, 
                s_input, s_w1, s_b1, s_w2, s_b2, s_w3, s_b3 );
    conv1(s_input, s_w1, s_b1, s_conv1);
    pool1(s_conv1, s_pool1);
    conv2(s_pool1, s_w2, s_b2, s_conv2);
    pool2(s_conv2, s_pool2);
    dense3(s_pool2, s_w3, s_b3, s_out);
    store_output(s_out, output);
}
