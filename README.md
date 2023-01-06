# trt2022_levit (For Demo)

## Overview

- Model name：LeViT， https://github.com/facebookresearch/LeViT. 
- We implement FP16 and INT8 optimization. After using Nsight System to analysize latency，we choose to write softmax customed plugin. Finally our model reach 1.7x acceleration compared to Pytorch CUDA.
- Model Name: LeViT, https://github.com/facebookresearch/LeViT

## Running Steps

- Environment Prepare：

Using TensorRT Docker image： 

```shell
docker run --gpus all -it --rm nvcr.io/nvidia/tensorrt:22.05-py3
```

clone code repo：

```shell
git clone https://github.com/ModelACC/trt2022_levit.git
```

Install Requirements：

```shell
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

- running softmax plugin building and testing script：
  
  ```shell
  cd LeViT 
  ./build_engine_run_plugin.sh
  ```

- Building TensorRT Engine：

```shell
cd LeViT
# Conver pytorch model to onnx model.
# MODEL can be one of 128S, 128, 192, 256, 384
# For example, 384 means LeViT-384
python3 trt_convert_onnx.py MODEL
# build TensorRT engine
python3 trt_build_engine.py --onnx-path ONNX_MODEL --engine-path ENGINE_PATH [--enable-fp16] [--enable-int8]
```

- Testing model accuracy

```shell
python3 valid.py \
--data-path IMAGENET_ROOT \
--model MODEL_NAME \
--type TYPE \
--engine-path ENGINE_PATH
```

IMAGENET_ROOT: imagenet dataset root path. The dataset should follow the format：

> IMAGENET_ROOT
> 
> ----| val
> 
> --------| n02089867
> 
> ------------| XXX.jpg
> 
> ------------| ......
> 
> --------| n02437616
> 
> ------------| YYY.jpg
> 
> ------------| ......
> 
> --------| ......

MODEL_NAME：The name of model, can be LeViT_384，LeViT_256, according to original repo for LeViT. 

TYPE： pytorch or tensorrt, representing PyTorch or TensorRT Model.

ENGINE_PATH：If TYPE is tensorrt, it need the engine file. 

- 测试模型推理速度

对于TensorRT模型，可以使用trtexec测速：

```shell
trtexec --loadEngine=ENGINE_PATH --shapes=input:16x3x224x224 --useSpinWait
```

对于PyTorch模型，使用python脚本文件测速：

```shell
python3 pytorch_speed_test.py
```

- 生成calibration数据集：

```shell
python3 generate_calibration_data.py --data-path IMAGENET_ROOT
```

## Original

### 1. Introduction

- Our group has chosen the LeViT model from Facebook Research as the optimized model for our use. LeViT is a hybrid model based on Transformer, designed for fast inference of image classification. It has achieved a good balance of both inference accuracy and running speed. Among models with similar accuracy, LeViT has a speed increase of almost five times compared to other state-of-the-art visual Transformer models such as Visual Transformer, Bottleneck transformer, and Pyramid vision transformer. In comparison to other Token-to-token ViT models, LeViT also has fewer parameters and FLOPs. Therefore, LeViT is a relatively lightweight model with good performance that ensures its practicality.

### 2. Model structure

- The overall structure of the model is shown in the following diagram. The input image is passed through pyramid convolution to extract the embedding, which is then input to three consecutive stages. The output of the final stage is used as input to the classifier and produces the image classification result.
- Before the data is input to the attention layer, it goes through a pyramid convolution process. This consists of four consecutive 3x3 convolution kernels, with the dimension of the output feature map increasing at each level. The purpose of this network structure is to extract patch embeddings and improve the accuracy of the network.
- In each stage, the input data passes through a series of multi-head attention layers and MLP layers for residual connections. The multi-head attention in LeViT is similar to the attention found in typical Transformer models. However, LeViT's attention layer does not use the common positional embedding. Instead, an attention bias is added to the QK product calculation to serve as a positional embedding. The MLP layer in this model consists of a 1x1 convolution layer and a batch normalization layer. According to the paper, the authors chose not to use a common MLP block because it typically has a larger computational cost, and the structure used in this model was adopted for the residual connection to reduce this cost.
- Additionally, between each pair of stages, there is a multi-head shrink attention layer. This layer reduces the width and height of the output feature map by half and increases its dimension, serving as a down-sampling operation. The structure of this layer is similar to the multi-head attention in LeViT, with the only difference being that the input is down-sampled before the Q transformation, resulting in the "shrink attention" effect.
    ![model fig](imgs/model_structure.png)

## Optimization Process

### 1. Nsight System Analysis

- As shown in the performance analysis graph and table below, during execution, the Softmax function consumes a significant amount of computational time in addition to matrix multiplication and convolution operations. Therefore, there is still room for optimization in this function, and we are attempting to optimize it by developing a TensorRT Plugin.
  ![Nsight system result](imgs/nsys_table.png)
  ![Nsight fig](imgs/nsys_fig.png)

### 2. Softmax Plugin 

- Softmax Kernel Function come from https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh 
- Performance improvement: The input shape of the Softmax function in the original model is [batch_size, 3, 196, 196], and we took batch_size = 8 for testing the original Softmax and the Softmax Plugin. Without losing accuracy, the running time of the Softmax Plugin is 61% of the original Softmax. For the LeViT model experiment, we used the LeViT-384 model for performance evaluation. The input shape of the model is [batch_size = 8, 3,224,224], and the performance improvement is approximately 5%.

### 3. INT8 Quantization

To achieve better acceleration, in addition to the conventional FP32 and FP16, we also quantized the model to INT8. We used the PTQ quantization process provided by TensorRT, which involves writing a calibrator manually and enabling INT8 quantization and passing in the calibrator when constructing the engine.

As expected, the model accuracy decreased by 16 points (from 82.6 to 66) after INT8 quantization. Based on past competition experience, the layers that are prone to problems after INT8 quantization are the Softmax, activation layers, and BN layers. LeViT has already fused BN and convolution when exporting to ONNX, so BN does not need to be considered. After experimentation, the Softmax layer had no effect on accuracy, and after manually not quantizing the Sigmoid activation function layer, the model's accuracy recovered to 76 (a recovery of 10 points).

## Accuracy and Optimization Result

All experiments were conducted on the A10 GPU. The batch size is 16, the size of a single image is 3x224x224, and there is warmup.

| LeViT-384       | acc@1 | latency (ms) | Acceleration Ratio |
| --------------- | ----- | ------------ | ------------------ |
| Pytorch (GPU)   | 82.6  | 8.91         | x1                 |
| TensorRT (FP32) | 82.6  | 8.28         | x1.08              |
| TensorRT (FP16) | 82.6  | 5.38         | x1.66              |
| TensorRT (INT8) | 76.0  | 5.05         | x1.76              |

| LeViT-256       | acc@1 | latency (ms) | Acceleration Ratio |
| --------------- | ----- | ------------ | ------------------ |
| Pytorch (GPU)   | 81.6  | 5.72         | x1                 |
| TensorRT (FP32) | 81.6  | 5.40         | x1.06              |
| TensorRT (FP16) | 81.5  | 3.42         | x1.67              |
| TensorRT (INT8) | 81.2  | 3.52         | x1.63              |
