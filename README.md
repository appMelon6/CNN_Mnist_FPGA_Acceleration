# CNN_MNIST_TOP – HLS CNN Accelerator Project

## 1. Project Overview

`CNN_MNIST_TOP` is a hardware accelerator implemented using Vivado HLS that performs handwritten digit classification on the MNIST dataset. The design implements a complete Convolutional Neural Network (CNN) in hardware using AXI-based memory interfaces and streaming dataflow architecture.

The implemented network structure is:

Input (28x28)  
→ Convolution Layer 1  
→ ReLU  
→ Max Pooling Layer 1  
→ Convolution Layer 2  
→ ReLU  
→ Max Pooling Layer 2  
→ Fully Connected (Dense) Layer  
→ Output (10 classes)

The accelerator is designed for deployment on FPGA platforms such as Pynq-Z2 and operates using fixed-point arithmetic.

---

## 2. Network Architecture

### 2.1 Input Layer
- Input size: 28 × 28
- Total elements: 784
- Data type: `ap_fixed<16,8>`

---

### 2.2 Convolution Layer 1 (conv1)
- Number of filters: 8
- Kernel size: 3 × 3
- Input channels: 1
- Output feature map size: 26 × 26 × 8
- Total weights: 8 × 3 × 3 = 72
- Bias values: 8
- Activation function: ReLU

Output stream elements: 26 × 26 = 676 (each element is `vec8_f`)

---

### 2.3 Max Pooling Layer 1 (pool1)
- Pooling window: 2 × 2
- Input size: 26 × 26 × 8
- Output size: 13 × 13 × 8

Output stream elements: 13 × 13 = 169 (`vec8_f`)

---

### 2.4 Convolution Layer 2 (conv2)
- Number of filters: 16
- Kernel size: 3 × 3
- Input channels: 8
- Output feature map size: 11 × 11 × 16
- Total weights: 16 × 8 × 3 × 3 = 1152
- Bias values: 16
- Activation function: ReLU

Output stream elements: 11 × 11 = 121 (`vec16_f`)

---

### 2.5 Max Pooling Layer 2 (pool2)
- Pooling window: 2 × 2
- Input size: 11 × 11 × 16
- Output size: 5 × 5 × 16

Output stream elements: 5 × 5 = 25 (`vec16_f`)

---

### 2.6 Fully Connected Layer (dense3)
- Input size: 16 × 5 × 5 = 400
- Output neurons: 10
- Weight matrix size: 10 × 400 = 4000
- Bias values: 10
- Output: 10 logits

Note: Softmax activation is not implemented in hardware. It may be applied in software if required.

---

## 3. Top-Level Function Interface

```cpp
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
```

---

## 4. AXI Interface Configuration

All large arrays are connected using AXI4-Master interfaces. Control signals use AXI4-Lite.

| Port     | Depth | Description |
|----------|--------|------------|
| input    | 784    | Input image data |
| weight1  | 72     | Conv1 weights |
| bias1    | 8      | Conv1 bias |
| weight2  | 1152   | Conv2 weights |
| bias2    | 16     | Conv2 bias |
| weight3  | 4000   | Dense weights |
| bias3    | 10     | Dense bias |
| output   | 10     | Output logits |

---

## 5. Streaming and Dataflow Architecture

The design uses `#pragma HLS DATAFLOW` to enable concurrent execution of layers. Intermediate feature maps are passed through HLS streams:

- `vec8_f` for 8-channel parallel feature maps
- `vec16_f` for 16-channel parallel feature maps

Processing pipeline:

conv1 → pool1 → conv2 → pool2 → dense3

All stages must produce and consume the exact number of stream elements to avoid deadlock.

---

## 6. Fixed-Point Configuration

- Data type: `ap_fixed<16,8>`
- Integer bits: 8
- Fractional bits: 8
- Scaling factor: 2^8 = 256

Conversion in software:

```python
x_fixed = round(x_float * 256)
x_float = x_fixed / 256
```

---

## 7. Feature Map Size Summary

| Layer  | Output Shape | Stream Elements |
|--------|--------------|-----------------|
| Conv1  | 26 × 26 × 8  | 676 vec8_f |
| Pool1  | 13 × 13 × 8  | 169 vec8_f |
| Conv2  | 11 × 11 × 16 | 121 vec16_f |
| Pool2  | 5 × 5 × 16   | 25 vec16_f |
| Dense  | 10           | 10 |

---

## 8. Execution Flow

1. Load input image and trained weights from DDR memory.
2. Execute streaming CNN pipeline using dataflow.
3. Store final output logits to DDR.
4. Software reads output and determines predicted class.

---

## 9. Verification Methodology

1. Train CNN model in TensorFlow.
2. Export weights and biases as `.npy` files.
3. Convert floating-point values to fixed-point representation.
4. Transfer data to FPGA via AXI.
5. Compare FPGA logits against TensorFlow logits.

---

## 10. Performance Target

- Target board: Pynq-Z2
- Target frequency: 50 MHz
- Streaming architecture with pipelined loops
- Designed for single-image inference acceleration

---

## 11. Notes

- Dense layer input size must be exactly 400.
- Stream element counts must match exactly between stages.
- Incorrect AXI address configuration may cause the accelerator to stall.
- Softmax is optional and typically performed in software.

---

End of Document
