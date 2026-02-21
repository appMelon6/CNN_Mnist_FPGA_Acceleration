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
);
