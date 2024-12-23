<img src="https://raw.githubusercontent.com/ChanLumerico/luma/main/img/title/nn_dark.png" alt="logo" height="50%" width="50%">

Deep learning models and neural network utilities of Luma

---

## Neural Layers

*luma.neural.layer 🔗*

### Convolution

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| `Conv1D` | $(N,C_{in},W)$ | $(N,C_{out},W_{pad})$ |
| `Conv2D` | $(N,C_{in},H,W)$ | $(N,C_{out},H_{pad},W_{pad})$ |
| `Conv3D` | $(N,C_{in},D,H,W)$ | $(N,C_{out},D_{pad},H_{pad},W_{pad})$ |
| `DepthConv1D` | $(N,C,W)$ | $(N,C,W_{pad})$ |
| `DepthConv2D` | $(N,C,H,W)$ | $(N,C,H_{pad},W_{pad})$ |
| `DepthConv3D` | $(N,C,D,H,W)$ | $(N,C,D_{pad},H_{pad},W_{pad})$ |

### Pooling

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| `Pool1D` | $(N,C,W_{in})$ | $(N,C,W_{out})$ |
| `Pool2D` | $(N,C,H_{in},W_{in})$ | $(N,C,H_{out},W_{out})$ |
| `Pool3D` | $(N,C,D_{in},H_{in},W_{in})$ | $(N,C,D_{out},H_{out},W_{out})$ |
| `GlobalAvgPool1D` | $(N,C,W)$ | $(N,C,1)$ |
| `GlobalAvgPool2D` | $(N,C,H,W)$ | $(N,C,1,1)$ |
| `GlovalAvgPool3D` | $(N,C,D,H,W)$ | $(N,C,1,1,1)$ |
| `AdaptiveAvgPool1D` | $(N,C,W_{in})$ | $(N,C,W_{out})$ |
| `AdaptiveAvgPool2D` | $(N,C,H_{in},W_{in})$ | $(N,C,H_{out},W_{out})$ |
| `AdaptiveAvgPool3D` | $(N,C,D_{in},H_{in},W_{in})$ | $(N,C,D_{out},H_{out},W_{out})$ |
| `LpPool1D` | $(N,C,W_{in})$ | $(N,C,W_{out})$ |
| `LpPool2D` | $(N,C,H_{in}, W_{in})$ | $(N,C,H_{out},W_{out})$ |
| `LpPool3D` | $(N,C,D_{in},H_{in},W_{in})$ | $(N,C,D_{out},H_{out},W_{out})$ |

### Dropout

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| `Dropout` | $(*)$ | $(*)$ |
| `Dropout1D` | $(N,C,W)$ | $(N,C,W)$ |
| `Dropout2D` | $(N,C,H,W)$ | $(N,C,H,W)$ |
| `Dropout3D` | $(N,C,D,H,W)$ | $(N,C,D,H,W)$ |
| `DropBlock1D` | $(N,C,W)$ | $(N,C,W)$ |
| `DropBlock2D` | $(N,C,H,W)$ | $(N,C,H,W)$ |
| `DropBlock3D` | $(N,C,D,H,W)$ | $(N,C,D,H,W)$ |

### Linear

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| `Flatten` | $(N, *)$ | $(N, -1)$ |
| `Reshape` | $(*_{in})$ | $(*_{out})$ |
| `Transpose` | $(*_{in})$ | $(*_{\mathcal{P}(in)})$ |
| `Dense` | $(N,L_{in})$ | $(N,L_{out})$ |
| `DenseND` | $(*,F_{in},*)$ | $(*,F_{out},*)$ |
| `Identity` | $(*)$ | $(*)$ |

### Normalization

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| `BatchNorm1D` | $(N,C,W)$ | $(N,C,W)$ |
| `BatchNorm2D` | $(N,C,H,W)$ | $(N,C,H,W)$ |
| `BatchNorm3D` | $(N,C,D,H,W)$ | $(N,C,D,H,W)$ |
| `LocalResponseNorm` | $(N,C,*)$ | $(N,C,*)$ |
| `GlobalResponseNorm` | $(N,C,*)$ | $(N,C,*)$ |
| `LayerNorm` | $(N,*)$ | $(N,*)$ |

### Attention

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| `ScaledDotProductAttention` | $(N,H,L,d_{head})$ | $(N,H,L,d_{head})$ |
| `MultiHeadAttention` | $(N,L,d_{model})$ | $(N,L,d_{model})$ |
| `CrossMultiHeadAttention` | $(N,L,d_{model})$ | $(N,L,d_{model})$ |

### Utility

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| `Slice` | $(*)$ | $(*_{sliced})$ |

---

## Neural Blocks

*luma.neural.block 🔗*

### Standard Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `ConvBlock1D` | 2~3 | $(N,C_{in},W_{in})$ | $(N,C_{out},W_{out})$ |
| `ConvBlock2D` | 2~3 | $(N,C_{in},H_{in}, W_{in})$ | $(N,C_{out},H_{out}, W_{out})$ |
| `ConvBlock3D` | 2~3 | $(N,C_{in},D_{in},H_{in},W_{in})$ | $(N,C_{out},D_{out},H_{out},W_{out})$ |
| `SeparableConv1D` | 3~5 | $(N,C_{in},W_{in})$ | $(N,C_{out},W_{out})$ |
| `SeparableConv2D` | 3~5 | $(N,C_{in},H_{in}, W_{in})$ | $(N,C_{out},H_{out}, W_{out})$ |
| `SeparableConv3D` | 3~5 | $(N,C_{in},D_{in},H_{in},W_{in})$ | $(N,C_{out},D_{out},H_{out},W_{out})$ |
| `DenseBlock` | 2~3 | $(N,L_{in})$ | $(N,L_{out})$ |

### Inception Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `IncepBlock.V1` | 19 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `IncepBlock.V2_TypeA` | 22 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `IncepBlock.V2_TypeB` | 31 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `IncepBlock.V2_TypeC` | 28 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `IncepBlock.V2_Redux` | 16 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `IncepBlock.V4_Stem` | 38 | $(N,3,299,299)$ | $(N,384,35,35)$ |
| `IncepBlock.V4_TypeA` | 24 | $(N,384,35,35)$ | $(N,384,35,35)$ |
| `IncepBlock.V4_TypeB` | 33 | $(N,1024,17,17)$ | $(N,1024,17,17)$ |
| `IncepBlock.V4_TypeC` | 33 | $(N,1536,8,8)$ | $(N,1536,8,8)$ |
| `IncepBlock.V4_ReduxA` | 15 | $(N,384,35,35)$ | $(N,1024,17,17)$ |
| `IncepBlock.V4_ReduxB` | 21 | $(N,1024,17,17)$ | $(N,1536,8,8)$ |

### Inception-Res Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `IncepResBlock.V1_TypeA` | 22 | $(N,256,35,35)$ | $(N,256,35,35)$ |
| `IncepResBlock.V1_TypeB` | 16 | $(N,896,17,17)$ | $(N,896,17,17)$ |
| `IncepResBlock.V1_TypeC` | 16 | $(N,1792,8,8)$ | $(N,1792,8,8)$ |
| `IncepResBlock.V1_Redux` | 24 | $(N,896,17,17)$ | $(N,1792,8,8)$ |
| `IncepResBlock.V2_TypeA` | 22 | $(N,384,35,35)$ | $(N,384,35,35)$ |
| `IncepResBlock.V2_TypeB` | 16 | $(N,1280,17,17)$ | $(N,1280,17,17)$ |
| `IncepResBlock.V2_TypeC` | 16 | $(N,2272,8,8)$ | $(N,2272,8,8)$ |
| `IncepResBlock.V2_Redux` | 24 | $(N,1280,17,17)$ | $(N,2272,8,8)$ |

### ResNet Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `ResNetBlock.Basic` | 7~ | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `ResNetBlock.Bottleneck` | 10~ | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `ResNetBlock.PreActBottleneck` | 10~ | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `ResNetBlock.Bottleneck_SE` | 16~ | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |

### Xception Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `XceptionBlock.Entry` | 42 | $(N,3,299,299)$ | $(N,728,19,19)$ |
| `XceptionBlock.Middle` | 14 | $(N,728,19,19)$ | $(N,728,19,19)$ |
| `XceptionBlock.Exit` | 11 | $(N,728,19,19)$ | $(N,1024,9,9)$ |

### SE(Squeeze & Excitation) Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `SEBlock1D` | 6 | $(N,C,W)$ | $(N,C,W)\text{ or }(N,C)$ |
| `SEBlock2D` | 6 | $(N,C,H,W)$ | $(N,C,H,W)\text{ or }(N,C)$ |
| `SEBlock3D` | 6 | $(N,C,D,H,W)$ | $(N,C,D,H,W)\text{ or }(N,C)$ |

### MobileNet Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `MobileNetBlock.InvRes` | 6~9 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `MobileNetBlock.InvRes_SE` | 14~17 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |

### DenseNet Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `DenseNetBlock.Composite` | 6 | $(N,C,H,W)$ | $(N,G,H,W)$ |
| `DenseNetBlock.DenseUnit` | $6\times l$ | $(N,C,H,W)$ | $(N,C+L\times G,H,W)$ |
| `DenseNetBlock.Transition` | 4 | $(N,C,H,W)$ | $(N,\lfloor\theta\times C\rfloor,\lfloor H/2\rfloor,\lfloor W/2\rfloor)$ |

### EfficientNet Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `EfficientBlock.MBConv` | 14~17 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| `EfficientBlock.FusedMBConv` | 12~15 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |

### SK(Selective Kernel) Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `SKBlock1D` | $5k+8$ | $(N,C_{in},W)$ | $(N,C_{out},W)$ |
| `SKBlock2D` | $5k+8$ | $(N,C_{in},H,W)$ | $(N,C_{out},H,W)$ |
| `SKBlock3D` | $5k+8$ | $(N,C_{in},D,H,W)$ | $(N,C_{out},D,H,W)$ |

### ResNeSt Block

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `ResNeStBlock` | $22\sim 24+8r$ | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |

### ConvNeXt Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `ConvNeXtBlock.V1` | 7 | $(N,C,H,W)$ | $(N,C,H,W)$ |
| `ConvNeXtBlock.V2` | 8 | $(N,C,H,W)$ | $(N,C,H,W)$ |

### Transformer Blocks

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| `PositionwiseFeedForward` | 3 | $(N,L,d_{model})$ | $(N,L,d_{model})$ |
| `Encoder` | 11 | $(N,L,d_{model})$ | $(N,L,d_{model})$ |
| `Decoder` | 13 | $(N,L,d_{model})$ | $(N,L,d_{model})$ |
| `EncoderStack` | $11n+0\sim2$ | $(N,L,d_{model})$ | $(N,L,d_{model})$ |
| `DecoderStack` | $13n$ | $(N,L,d_{model})$ | $(N,L,d_{model})$ |

*Waiting for future updates…🔮*

---

## Neural Models

*luma.neural.model 🔗*

### Image Classification Models

#### LeNet Series

> LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `LeNet_1` | 6 | $(N,1,28,28)$ | 2,180 | 22 | 2,202 | ✅ |
| `LeNet_4` | 8 | $(N,1,32,32)$ | 50,902 | 150 | 51,052 | ✅ |
| `LeNet_5` | 10 | $(N,1,32,32)$ | 61,474 | 236 | 61,170 | ✅ |

#### AlexNet Series

> Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." Advances in Neural
Information Processing Systems, 2012.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `AlexNet` | 21 | $(N,3,227,227)$ | 62,367,776 | 10,568 | 62,378,344 | ✅ |
| `ZFNet` | 21 | $(N,3,227,227)$ | 58,292,000 | 9,578 | 58,301,578 | ✅ |

#### VGGNet Series

> Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `VGGNet_11` | 27 | $(N,3,224,224)$ | 132,851,392 | 11,944 | 132,863,336 | ✅ |
| `VGGNet_13` | 31 | $(N,3,224,224)$ | 133,035,712 | 12,136 | 133,047,848 | ✅ |
| `VGGNet_16` | 37 | $(N,3,224,224)$ | 138,344,128 | 13,416 | 138,357,544 | ✅ |
| `VGGNet_19` | 43 | $(N,3,224,224)$ | 143,652,544 | 14,696 | 143,667,240 | ✅ |

#### Inception Series

*Inception-v1, v2, v3*

> Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1-9.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `Inception_V1` | 182 | $(N,3,224,224)$ | 6,990,272 | 8,280 | 6,998,552 | ✅ |
| `Inception_V2` | 242 | $(N,3,299,299)$ | 24,974,688 | 20,136 | 24,994,824 | ✅ |
| `Inception_V3` | 331 | $(N,3,299,299)$ | 25,012,960 | 20,136 | 25,033,096 | ✅ |

*Inception-v4, Inception-ResNet-v1, v2*

> Szegedy, Christian, et al. “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.” Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, 2017, pp. 4278-4284.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `Inception_V4` | 504 | $(N,3,299,299)$ | 42,641,952 | 32,584 | 42,674,536 | ✅ |
| `Inception_ResNet_V1` | 410 | $(N,3,299,299)$ | 21,611,648 | 33,720 | 21,645,368 | ✅ |
| `Inception_ResNet_V2` | 431 | $(N,3,299,299)$ | 34,112,608 | 43,562 | 34,156,170 | ✅ |

*Xception*

> Chollet, François. “Xception: Deep Learning with Depthwise Separable Convolutions.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1251-1258.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `Xception` | 174 | $(N,3,299,299)$ | 22,113,984 | 50,288 | 22,164,272 | ✅ |

#### ResNet Series

*ResNet-18, 34, 50, 101, 152*

> He, Kaiming, et al. “Deep Residual Learning for Image Recognition.“ Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `ResNet_18` | 77 | $(N,3,224,224)$ | 11,688,512 | 5,800 | 11,694,312 | ✅ |
| `ResNet_34` | 149 | $(N,3,224,224)$ | 21,796,672 | 9,512 | 21,806,184 | ✅ |
| `ResNet_50` | 181 | $(N,3,224,224)$ | 25,556,032 | 27,560 | 25,583,592 | ✅ |
| `ResNet_101` | 367 | $(N,3,224,224)$ | 44,548,160 | 53,762 | 44,601,832 | ✅ |
| `ResNet_152` | 554 | $(N,3,244,244)$ | 60,191,808 | 76,712 | 60,268,520 | ✅ |

*ResNet-200, 269, 1001*

> He, Kaiming, et al. “Identity Mappings in Deep Residual Networks.” European Conference on Computer Vision (ECCV), 2016, pp. 630-645.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `ResNet_200` | 793 | $(N,3,244,244)$ | 64,668,864 | 89,000 | 64,757,864 | ✅ |
| `ResNet_269` | 1,069 | $(N,3,244,244)$ | 102,068,416 | 127,400 | 102,195,816 | ✅ |
| `ResNet_1001` | 1,657 | $(N,3,224,224)$ | 159,884,992 | 208,040 | 160,093,032 | ✅ |

#### MobileNet Series

*MobileNet-v1*

> Howard, Andrew G., et al. “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.” arXiv, 17 Apr. 2017, [arxiv.org/abs/1704.04861](http://arxiv.org/abs/1704.04861).
> 

*MobileNet-v2*

> Sandler, Mark, et al. “MobileNetV2: Inverted Residuals and Linear Bottlenecks.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510-4520.
> 

*MobileNet-v3 Small, Large*

> Howard, Andrew, et al. “Searching for MobileNetV3.” Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1314-1324.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `MobileNet_V1` | 31 | $(N,3,224,224)$ | 4,230,976 | 11,944 | 4,242,920 | ✅ |
| `MobileNet_V2` | 148 | $(N,3,224,224)$ | 8,418,624 | 19,336 | 8,437,960 | ✅ |
| `MobileNet_V3_S` | 161 | $(N,3,224,224)$ | 32,455,856 | 326,138 | 32,781,994 | ✅ |
| `MobileNet_V3_L` | 180 | $(N,3,224,224)$ | 167,606,960 | 1,136,502 | 168,743,462 | ✅ |

#### SENet Series

> Hu, Jie, et al. “Squeeze-and-Excitation Networks.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132-7141.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `SE_ResNet_50` | 263 | $(N,3,224,224)$ | 35,615,808 | 46,440 | 35,662,248 | ✅ |
| `SE_ResNet_152` | 803 | $(N,3,224,224)$ | 86,504,512 | 136,552 | 86,641,064 | ✅ |
| `SE_Inception_ResNet_V2` | 569 | $(N,3,299,299)$ | 58,794,080 | 80,762 | 58,874,842 | ✅ |
| `SE_DenseNet_121` | 388 | $(N,3,224,224)$ | 9,190,272 | 14,760 | 9,205,032 | ✅ |
| `SE_DenseNet_169` | 532 | $(N,3,224,224)$ | 16,515,968 | 19,848 | 16,535,816 | ✅ |
| `SE_ResNeXt_50` | 299 | $(N,3,224,224)$ | 37,135,680 | 53,992 | 37,189,672 | ✅ |
| `SE_ResNeXt_101` | 605 | $(N,3,224,224)$ | 65,197,376 | 110,568 | 65,307,944 | ✅ |

#### DenseNet Series

> Huang, Gao, et al. "Densely Connected Convolutional Networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 4700-4708.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `DenseNet_121` | 364 | $(N,3,224,224)$ | 7,977,856 | 11,240 | 7,989,096 | ✅ |
| `DenseNet_169` | 508 | $(N,3,224,224)$ | 14,148,480 | 15,208 | 14,163,688 | ✅ |
| `DenseNet_201` | 604 | $(N,3,299,299)$ | 20,012,928 | 18,024 | 20,030,952 | ✅ |
| `DenseNet_264` | 794 | $(N,3,299,299)$ | 33,336,704 | 23,400 | 33,360,104 | ✅ |

#### EfficientNet Series

*EfficientNet-B0, B1, B2, B3, B4, B5, B6, B7*

> Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." International Conference on Machine Learning, 2020, pp. 6105-6114. arXiv:1905.11946.
>

*EfficientNet-v2-S, M, L, XL*

>Tan, Mingxing, and Quoc Le. “EfficientNetV2: Smaller Models and Faster Training.” Proceedings of the 38th International Conference on Machine Learning (ICML), vol. 139, 2021, pp. 10096-10106.
>

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `EfficientNet_B0` | 262 | $(N,3,224,224)$ | 4,803,040 | 24,268 | 4,827,308 | ✅ |
| `EfficientNet_B1` | 310 | $(N,3,240,240)$ | 6,544,500 | 32,568 | 6,577,068 | ✅ |
| `EfficientNet_B2` | 358 | $(N,3,260,260)$ | 8,503,007 | 40,160 | 8,543,167 | ✅ |
| `EfficientNet_B3` | 435 | $(N,3,300,300)$ | 13,657,980 | 57,390 | 13,715,370 | ✅ |
| `EfficientNet_B4` | 515 | $(N,3,380,380)$ | 17,877,155 | 72,278 | 17,949,433 | ✅ |
| `EfficientNet_B5` | 611 | $(N,3,456,456)$ | 24,674,011 | 94,261 | 24,768,272 | ✅ |
| `EfficientNet_B6` | 768 | $(N,3,528,528)$ | 38,260,230 | 132,704 | 38,392,934 | ✅ |
| `EfficientNet_B7` | 925 | $(N,3,600,600)$ | 56,528,906 | 178,066 | 56,706,972 | ✅ |
| `EfficientNet_V2_S` | 634 | $(N,3,384,384)$ | 18,414,552 | 86,116 | 18,500,668 | ✅ |
| `EfficientNet_V2_M` | 864 | $(N,3,480,480)$ | 46,012,920 | 162,264 | 46,175,184 | ✅ |
| `EfficientNet_V2_L` | 1,292 | $(N,3,480,480)$ | 104,084,896 | 303,032 | 104,387,928 | ✅ |
| `EfficientNet_V2_XL` | 1,530 | $(N,3,480,480)$ | 201,762,976 | 450,227 | 202,213,203 | ✅ |

#### ResNeXt Series

*ResNeXt-50, 101*

>Xie, Saining, et al. "Aggregated Residual Transformations for Deep Neural Networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1492-1500.
>

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `ResNeXt_50` | 187 | $(N,3,224,224)$ | 25,027,904 | 35,112 | 25,063,016 | ✅ |
| `ResNeXt_101` | 374 | $(N,3,224,224)$ | 44,176,704 | 69,928 | 44,246,632 | ✅ |

#### SKNet Series

>Li, Xiang, et al. "Selective Kernel Networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 510-519.
>

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `SK_ResNet_50` | 443 | $(N,3,224,224)$ | 57,236,160 | 39,124 | 57,275,284 | ✅ |
| `SK_ResNet_101` | 902 | $(N,3,224,224)$ | 104,298,688 | 78,564 | 104,377,252 | ✅ |
| `SK_ResNeXt_50` | 443 | $(N,3,224,224)$ | 29,915,712 | 58,240 | 29,973,952 | ✅ |
| `SK_ResNeXt_101` | 902 | $(N,3,224,224)$ | 53,399,104 | 119,712 | 53,518,816 | ✅ |

#### ResNeSt Series

>Zhang, Hang, et al. “ResNeSt: Split-Attention Networks.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 2736-2746.
>

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `ResNeSt_50` | 517 | $(N,3,224,224)$ | 26,535,136 | 39,944 | 26,575,080 | ✅ |
| `ResNeSt_101` | 1,044 | $(N,3,224,224)$ | 46,371,552 | 80,200 | 46,451,752 | ✅ |
| `ResNeSt_200` | 2,067 | $(N,3,224,224)$ | 67,392,736 | 134,664 | 67,527,400 | ✅ |
| `ResNeSt_269` | 2,780 | $(N,3,224,224)$ | 106,451,680 | 193,864 | 106,645,544 | ✅ |

#### ConvNeXt Series

*ConvNeXt-v1*

>Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11976-11986.
>

*ConvNeXt-v2*

>Zhou, Xinyu, et al. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." arXiv preprint arXiv:2301.00808 (2023).
>

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `ConvNeXt_T` | 137 | $(N,3,224,224)$ | 28,524,000 | 42,184 | 28,566,184 | ✅ |
| `ConvNeXt_S` | 263 | $(N,3,224,224)$ | 50,096,352 | 83,656 | 50,180,008 | ✅ |
| `ConvNeXt_B` | 263 | $(N,3,224,224)$ | 88,422,016 | 111,208 | 88,533,224 | ✅ |
| `ConvNeXt_L` | 263 | $(N,3,224,224)$ | 197,513,664 | 166,312 | 197,679,976 | ✅ |
| `ConvNeXt_XL` | 263 | $(N,3,224,224)$ | 349,859,072 | 221,416 | 350,080,488 | ✅ |
| `ConvNeXt_V2_A` | 108 | $(N,3,224,224)$ | 3,690,800 | 12,640 | 3,703,440 | ✅ |
| `ConvNeXt_V2_F` | 108 | $(N,3,224,224)$ | 5,212,320 | 14,968 | 5,227,288 | ✅ |
| `ConvNeXt_V2_P` | 108 | $(N,3,224,224)$ | 9,038,720 | 19,624 | 9,058,344 | ✅ |
| `ConvNeXt_V2_N` | 124 | $(N,3,224,224)$ | 15,584,480 | 28,120 | 15,612,600 | ✅ |
| `ConvNeXt_V2_T` | 156 | $(N,3,224,224)$ | 28,576,992 | 42,184 | 28,619,176 | ✅ |
| `ConvNeXt_V2_B` | 300 | $(N,3,224,224)$ | 88,566,400 | 111,208 | 88,677,608 | ✅ |
| `ConvNeXt_V2_L` | 300 | $(N,3,224,224)$ | 197,730,240 | 166,312 | 197,896,552 | ✅ |
| `ConvNeXt_V2_H` | 300 | $(N,3,224,224)$ | 659,875,040 | 304,072 | 660,179,112 | ✅ |

#### CoAtNet Series

*Waiting for future updates…🔮*

### Sequence-to-Sequence Models

#### Transformer Series

*Transformer-(Base, Big)*

>Vaswani, Ashish, et al. "Attention Is All You Need." Advances in Neural Information Processing Systems, vol. 30, 2017.
>

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| `Transformer_Base` | 147 | $(N,L,512)$ | 62,984,192 | 104,584 | 63,088,776 | ✅ |
| `Transformer_Big` | 147 | $(N,L,1024)$ | 214,048,768 | 172,168 | 214,220,936 | ✅ |

*Waiting for future updates…🔮*

---

## How to Use `NeuralModel`

*luma.neural.base.NeuralModel 🔗*

The class `NeuralModel` is an abstract base class(ABC) for neural network models, supporting customized neural networks and dynamic model construction.

### 1️⃣ Create a new model class

Create a class for your custom neural model, inheriting `NeuralModel`.

```python
class MyModel(NeuralModel): ...
```

### 2️⃣ Build a constructor method

Add a constructor method `__init__` with all the necessary arguments for `NeuralModel` included. You can add additional arguments if needed. All the necessary arguments will be auto-completed.

```python
class MyModel(NeuralModel):
    def __init__(
        self,
        batch_size: int,
        n_epochs: int,
        valid_size: float,
        early_stopping: bool,
        patience: int,
        shuffle: bool,
        random_state: int | None,
        deep_verbose: bool,
    ) -> None:
        ...
```

### 3️⃣ Initialize `self.model` attribute as a new instance of `Sequential`

Your new custom neural model’s components(i.e. layers, blocks) are stacked up at the inherited `self.model`. 

```python
def __init__(self, ...) -> None:
    ...
    self.model = Sequential(
        # Insert Layers if needed
    )
```

### 4️⃣ Call `init_model()` to initialize the model attribute

You must call the inherited method `init_model()` inside `__init__` in order to initialize `self.model`.

```python
def __init__(self, ...) -> None:
    ...
    self.model = Sequential(...)
    self.init_model()  # Necessary
```

### 5️⃣ Call `build_model()` to construct the neural network model

You must call the inherited abstract method `build_model()` inside `__init__`  in order to construct the neural network.

```python
def __init__(self, ...) -> None:
    ...
    self.model = Sequential(...)
    self.init_model()
    self.build_model()  # Necessary
```

### 6️⃣ Implement `build_model()`

Use `Sequential`’s methods(i.e. `add()`, `extend()`) to build your custom neural network. 

```python
def build_method(self) -> None:
    self.model.add(
        Conv2D(3, 64, 3),
    )
    self.model.extend(
        Conv2D(64, 64, 3),
        BatchNorm2D(64),
        Activation.ReLU(),
    )
    ...  # Additional Layers
```

### 7️⃣ Set optimizer, loss function, and LR scheduler

An optimizer and a loss function must be assigned to the model. Learning Rate(LR) scheduler is optional.

```python
model = MyModel(...)

# Necessary
model.set_optimizer(AdamOptimizer(...))
model.set_loss(CrossEntropy())

# Optional
model.set_lr_scheduler(ReduceLROnPlateau(...))
```

### 8️⃣ Fit the model

Use `NeuralModel`’s method `fit_nn()` to train the model with a train dataset.

```python
model.fit_nn(X_train, y_train)
```

### 9️⃣ Make predictions

Use `NeuralModel`’s method `predict_nn()` to predict an unseen dataset.

```python
y_pred = model.predict_nn(X_test, argmax=True)
```

### See Also

For more detailed information, please refer to the source code of `NeuralModel`.

## Recommendations

Since Luma's neural package supports `MLX` acceleration, platforms with **Apple Silicon** are recommended.

Apple's Metal Performance Shader(MPS) will be automatically detected if available, and `MLX` acceleration will be applied.

Otherwise, the default CPU operations are applied.