#include <iostream>
#include "data_types.h" // Ensure this defines data_f and your vector types
#include "CNN_MNIST_TOP.h"

// Prototype of your top function
void CNN_MNIST_TOP(
    data_f *input,
    data_f *weight1, data_f *bias1,
    data_f *weight2, data_f *bias2,
    data_f *weight3, data_f *bias3,
    data_f *output
);

int main() {
    // 1. Allocate memory for inputs, weights, and outputs
    data_f input[784];       // 28x28
    data_f weight1[72];      // 8*3*3
    data_f bias1[8];
    data_f weight2[1152];    // 16*8*3*3
    data_f bias2[16];
    data_f weight3[4000];    // 10*400
    data_f bias3[10];
    data_f output[10];

    // 2. Initialize with dummy data (In a real scenario, load these from .bin or .txt)
    for(int i = 0; i < 784; i++) input[i] = (data_f)0.5;   // Semi-gray image
    for(int i = 0; i < 72;   i++) weight1[i] = (data_f)0.01;
    for(int i = 0; i < 4000; i++) weight3[i] = (data_f)0.001;
    for(int i = 0; i < 10;   i++) bias3[i] = (data_f)0.1;

    std::cout << ">> Starting Simulation..." << std::endl;

    // 3. Call the Hardware Top Function
    CNN_MNIST_TOP(input, weight1, bias1, weight2, bias2, weight3, bias3, output);

    // 4. Check Results
    std::cout << ">> Classification Results (Softmax/Logits):" << std::endl;
    int predicted_digit = 0;
    float max_val = -1e10;

    for (int i = 0; i < 10; i++) {
        float val = (float)output[i];
        std::cout << "Digit [" << i << "]: " << val << std::endl;
        
        if (val > max_val) {
            max_val = val;
            predicted_digit = i;
        }
    }

    std::cout << ">> Predicted Digit: " << predicted_digit << std::endl;

    // 5. Validation (Simple check to ensure it's not all zeros)
    if (max_val == 0) {
        std::cout << "!! TEST FAILED: Output is all zeros." << std::endl;
        return 1;
    }

    std::cout << ">> TEST PASSED" << std::endl;
    return 0;
}