#pragma once
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<16,8> data_f;
typedef ap_fixed<32,16> acc_f;
struct vec8_f {
    data_f ch[8];
};
struct vec16_f {
    data_f ch[16];
};