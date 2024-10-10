"""
Image Classification Models
---------------------------

This module contains a collection of neural network architectures 
commonly used for image classification tasks. These models are designed 
to process and classify images into various categories based on their features.

The models differ in complexity, depth, and computational efficiency, making 
them suitable for a wide range of applications, from resource-constrained 
environments to large-scale, high-accuracy image recognition systems.

The models included in this module cover a variety of design paradigms, 
such as convolutional neural networks (CNNs), residual networks, 
and Inception mechanisms, ensuring flexibility for various use cases.
"""

from . import (
    alex,
    convnext,
    dense,
    efficient,
    incep,
    lenet,
    mobile,
    resnest,
    resnet,
    resnext,
    vgg,
)


__all__ = (
    "LeNet_1",
    "LeNet_4",
    "LeNet_5",
    "AlexNet",
    "ZFNet",
    "VGGNet_11",
    "VGGNet_13",
    "VGGNet_16",
    "VGGNet_19",
    "Inception_V1",
    "Inception_V2",
    "Inception_V3",
    "Inception_V4",
    "Inception_ResNet_V1",
    "Inception_ResNet_V2",
    "ResNet_18",
    "ResNet_34",
    "ResNet_50",
    "ResNet_101",
    "ResNet_152",
    "ResNet_200",
    "ResNet_269",
    "ResNet_1001",
    "Xception",
    "MobileNet_V1",
    "MobileNet_V2",
    "MobileNet_V3_S",
    "MobileNet_V3_L",
    "SE_ResNet_50",
    "SE_ResNet_152",
    "SE_Inception_ResNet_V2",
    "SE_DenseNet_121",
    "SE_DenseNet_169",
    "SE_ResNeXt_50",
    "SE_ResNeXt_101",
    "DenseNet_121",
    "DenseNet_169",
    "DenseNet_201",
    "DenseNet_264",
    "EfficientNet_B0",
    "EfficientNet_B1",
    "EfficientNet_B2",
    "EfficientNet_B3",
    "EfficientNet_B4",
    "EfficientNet_B5",
    "EfficientNet_B6",
    "EfficientNet_B7",
    "EfficientNet_V2_S",
    "EfficientNet_V2_M",
    "EfficientNet_V2_L",
    "EfficientNet_V2_XL",
    "ResNeXt_50",
    "ResNeXt_101",
    "SK_ResNet_50",
    "SK_ResNet_101",
    "SK_ResNeXt_50",
    "SK_ResNeXt_101",
    "ResNeSt_50",
    "ResNeSt_101",
    "ResNeSt_200",
    "ResNeSt_269",
    "ConvNeXt_T",
    "ConvNeXt_S",
    "ConvNeXt_B",
    "ConvNeXt_L",
    "ConvNeXt_XL",
)


class LeNet_1(lenet._LeNet_1):
    """
    LeNet-1 is an early convolutional neural network (CNN) proposed by
    Yann LeCun in 1988, primarily designed for handwritten character
    recognition. It consists of two convolutional layers interleaved
    with subsampling layers, followed by a fully connected layer.
    The network uses convolutions to automatically learn spatial
    hierarchies of features, which are then used for classification
    tasks. LeNet-1 was one of the first successful applications of CNNs,
    laying the groundwork for more complex architectures in image
    processing.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 1, 28, 28] -> Matrix[-1, 10]
    ```
    Parameter Size:
    ```txt
    2,180 weights, 22 biases -> 2,202 params
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Tanh
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=10
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.

    """


class LeNet_4(lenet._LeNet_4):
    """
    LeNet-4 is a specific convolutional neural network structure designed
    for more advanced image recognition tasks than its predecessors.
    This version incorporates several layers of convolutions and pooling,
    followed by fully connected layers leading to the output for classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 1, 32, 32] -> Matrix[-1, 10]
    ```
    Parameter Size:
    ```txt
    50,902 weights, 150 biases -> 51,052 params
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Tanh
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=10
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
    """


class LeNet_5(lenet._LeNet_5):
    """
    LeNet-5 is a specific convolutional neural network structure designed
    for more advanced image recognition tasks than its predecessors.
    This version incorporates several layers of convolutions and pooling,
    followed by fully connected layers leading to the output for classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 1, 32, 32] -> Matrix[-1, 10]
    ```
    Parameter Size:
    ```txt
    61,474 weights, 236 biases -> 61,710 params
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Tanh
        Type of activation function
        Type of loss function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=10
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip
    Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
    """


class AlexNet(alex._AlexNet):
    """
    AlexNet is a deep convolutional neural network that is designed for
    challenging image recognition tasks and was the winning entry in ILSVRC 2012.
    This architecture uses deep layers of convolutions with ReLU activations,
    max pooling, dropout, and fully connected layers leading to a classification
    output.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    62,367,776 weights, 10,568 biases -> 62,378,344 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet
    Classification with Deep Convolutional Neural Networks." Advances in Neural
    Information Processing Systems, 2012.

    """


class ZFNet(alex._ZFNet):
    """
    ZFNet is a refinement of the AlexNet architecture that was specifically
    designed to improve model understanding and performance on image recognition
    tasks. This model was presented by Matthew Zeiler and Rob Fergus in their
    paper and was particularly notable for its improvements in layer configurations
    that enhanced visualization of intermediate activations, aiding in understanding
    the functioning of deep convolutional networks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 227, 227] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    58,292,000 weights, 9,578 biases -> 58,301,578 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Zeiler, Matthew D., and Rob Fergus. "Visualizing and Understanding
    Convolutional Networks." European conference on computer vision, 2014.

    """


class VGGNet_11(vgg._VGGNet_11):
    """
    VGG11 is a simplified variant of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    132,851,392 weights, 11,944 biases -> 132,863,336 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """


class VGGNet_13(vgg._VGGNet_13):
    """
    VGG13 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    133,035,712 weights, 12,136 biases -> 133,047,848 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """


class VGGNet_16(vgg._VGGNet_16):
    """
    VGG16 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    138,344,128 weights, 13,416 biases -> 138,357,544 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """


class VGGNet_19(vgg._VGGNet_19):
    """
    VGG19 is ont of the variants of the VGG network architecture that was designed
    to enhance image recognition performance through deeper networks with smaller
    convolutional filters. This model was introduced by Karen Simonyan and Andrew
    Zisserman in their paper and is notable for its simplicity and effectiveness
    in image classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    143,652,544 weights, 14,696 biases -> 143,667,240 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvBlock2D(), DenseBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for
    Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
    """


class Inception_V1(incep._Inception_V1):
    """
    Inception v1, also known as GoogLeNet, is a deep convolutional neural network
    architecture designed for image classification. It introduces an "Inception
    module," which uses multiple convolutional filters of different sizes in
    parallel to capture various features at different scales. This architecture
    reduces computational costs by using 1x1 convolutions to decrease the number
    of input channels. Inception v1 achieved state-of-the-art results on the
    ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2014.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    6,990,272 weights, 8,280 biases -> 6,998,552 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V1()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings
    of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2015, pp. 1-9.
    """


class Inception_V2(incep._Inception_V2):
    """
    Inception v2, an improvement of the original Inception architecture,
    enhances computational efficiency and accuracy in deep learning models.
    It introduces the factorization of convolutions and additional
    normalization techniques to reduce the number of parameters and improve
    training stability. These modifications allow for deeper and more
    complex neural networks with improved performance.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    24,974,688 weights, 20,136 biases -> 24,994,824 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V2_TypeA(),
    IncepBlock.V2_TypeB(),
    IncepBlock.V2_TypeC(),
    IncepBlock.V2_Redux()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings
    of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2015, pp. 1-9.
    """


class Inception_V3(incep._Inception_V3):
    """
    Inception v3, an enhancement of Inception v2, further improves
    computational efficiency and accuracy in deep learning models.
    It includes advanced factorization of convolutions, improved grid
    size reduction techniques, extensive Batch Normalization, and
    label smoothing to prevent overfitting. These modifications enable
    deeper and more complex neural networks with significantly
    enhanced performance and robustness.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    25,012,960 weights, 20,136 biases -> 25,033,096 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V2_TypeA(),
    IncepBlock.V2_TypeB(),
    IncepBlock.V2_TypeC(),
    IncepBlock.V2_Redux()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings
    of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2015, pp. 1-9.
    """


class Inception_V4(incep._Inception_V4):
    """
    Inception v4, an enhancement of Inception v3, improves computational
    efficiency and accuracy. It includes sophisticated convolution
    factorization, refined grid size reduction, extensive Batch
    Normalization, and label smoothing. These advancements enable deeper
    and more robust neural networks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    42,641,952 weights, 32,584 biases -> 42,674,536 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepBlock.V4_Stem(),
    IncepBlock.V4_TypeA(),
    IncepBlock.V4_TypeB(),
    IncepBlock.V4_TypeC(),
    IncepBlock.V4_ReduxA(),
    IncepBlock.V4_ReduxB()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Szegedy, Christian, et al. “Inception-v4, Inception-ResNet and the
    Impact of Residual Connections on Learning.” Proceedings of the Thirty-First
    AAAI Conference on Artificial Intelligence, 2017, pp. 4278-4284.
    """


class Inception_ResNet_V1(incep._Inception_ResNet_V1):
    """
    Inception-ResNet v1 combines Inception modules with residual connections,
    improving computational efficiency and accuracy. This architecture uses
    convolution factorization, optimized grid size reduction, extensive
    Batch Normalization, and label smoothing, resulting in deeper and more
    robust neural networks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    21,611,648 weights, 33,720 biases -> 21,645,368 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepResBlock.V1_Stem(),
    IncepResBlock.V1_TypeA(),
    IncepResBlock.V1_TypeB(),
    IncepResBlock.V1_TypeC(),
    IncepResBlock.V1_Redux(),

    IncepBlock.V4_ReduxA()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """


class Inception_ResNet_V2(incep._Inception_ResNet_V2):
    """
    Inception-ResNet v2 enhances v1 with a deeper architecture and
    improved residual blocks for better performance. It features refined
    convolution factorization, more extensive Batch Normalization, and
    advanced grid size reduction.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    34,112,608 weights, 43,562 biases -> 34,156,170 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    IncepResBlock.V2_TypeA(),
    IncepResBlock.V2_TypeB(),
    IncepResBlock.V2_TypeC(),
    IncepResBlock.V2_Redux(),

    IncepBlock.V4_Stem(),
    IncepBlock.V4_ReduxA()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """


class ResNet_18(resnet._ResNet_18):
    """
    ResNet-18 is a 18-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    11,688,512 weights, 5,800 biases -> 11,694,312 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Basic()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """


class ResNet_34(resnet._ResNet_34):
    """
    ResNet-34 is a 34-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    21,796,672 weights, 9,512 biases -> 21,806,184 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Basic()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """


class ResNet_50(resnet._ResNet_50):
    """
    ResNet-50 is a 50-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    25,556,032 weights, 27,560 biases -> 25,583,592 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """


class ResNet_101(resnet._ResNet_101):
    """
    ResNet-101 is a 101-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    44,548,160 weights, 53,672 biases -> 44,601,832 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """


class ResNet_152(resnet._ResNet_152):
    """
    ResNet-152 is a 152-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    60,191,808 weights, 76,712 biases -> 60,268,520 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Deep Residual Learning for Image Recognition.”
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR), 2016, pp. 770-778.

    """


class ResNet_200(resnet._ResNet_200):
    """
    ResNet-200 is a 200-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    64,668,864 weights, 89,000 biases -> 64,757,864 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.PreActBottleneck()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Identity Mappings in Deep Residual Networks.”
    European Conference on Computer Vision (ECCV), 2016, pp. 630-645.

    """


class ResNet_269(resnet._ResNet_269):
    """
    ResNet-269 is a 269-layer deep neural network that uses residual blocks
    to improve training by learning residuals, helping prevent vanishing
    gradients and enabling better performance in image recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    102,068,416 weights, 127,400 biases -> 102,195,816 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.PreActBottleneck()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. He, Kaiming, et al. “Identity Mappings in Deep Residual Networks.”
    European Conference on Computer Vision (ECCV), 2016, pp. 630-645.

    """


class ResNet_1001(resnet._ResNet_1001):
    """
    ResNet-1001 is a 1001-layer deep neural network that uses residual
    blocks to improve training by learning residuals, helping prevent
    vanishing gradients and enabling better performance in image
    recognition tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    159,884,992 weights, 208,040 biases -> 160,093,032 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.PreActBottleneck()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    Warnings
    --------
    * This model has highly intensive depth of convolutional layers.
    Please consider your computational power, memory, etc.

    References
    ----------
    1. He, Kaiming, et al. “Identity Mappings in Deep Residual Networks.”
    European Conference on Computer Vision (ECCV), 2016, pp. 630-645.

    """


class Xception(incep._Xception):
    """
    Xception enhances the Inception architecture by replacing standard
    convolutions with depthwise separable convolutions, making it more
    efficient and effective at feature extraction. This design reduces
    the number of parameters and computations while maintaining or
    improving model accuracy on complex tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    22,113,984 weights, 50,288 biases -> 22,164,272 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    XceptionBlock.Entry(),
    XceptionBlock.Middle(),
    XceptionBlock.Exit(),

    SeparableConv2D()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Chollet, François. “Xception: Deep Learning with Depthwise
    Separable Convolutions.” Proceedings of the IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), 2017, pp.
    1251-1258.

    """


class MobileNet_V1(mobile._Mobile_V1):
    """
    MobileNet-V1 uses depthwise separable convolutions to significantly
    reduce the number of parameters and computational cost, making it
    highly efficient for mobile and embedded devices. It balances
    accuracy and efficiency through adjustable width and resolution
    multipliers.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    4,230,976 weights, 11,944 biases -> 4,242,920 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    SeparableConv2D()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `width_param` : float, default=1.0
        Width parameter(alpha) of the network
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Howard, Andrew G., et al. “MobileNets: Efficient Convolutional
    Neural Networks for Mobile Vision Applications.” arXiv preprint
    arXiv:1704.04861 (2017).

    """


class MobileNet_V2(mobile._Mobile_V2):
    """
    MobileNet-V2 builds on the efficiency of its predecessor by introducing
    inverted residuals and linear bottlenecks, further reducing
    computational cost and enhancing performance on mobile and embedded
    devices. It continues to balance accuracy and efficiency while allowing
    for flexible adjustments through width and resolution multipliers.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    8,418,624 weights, 19,336 biases -> 8,437,960 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    MobileNetBlock.InvRes()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU6
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `width_param` : float, default=1.0
        Width parameter(alpha) of the network
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Sandler, Mark, et al. “MobileNetV2: Inverted Residuals and Linear
    Bottlenecks.” Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 4510-4520.

    """


class MobileNet_V3_S(mobile._MobileNet_V3_S):
    """
    MobileNet-V3-Small improves on its predecessors by incorporating
    squeeze-and-excitation (SE) modules and the hard-swish activation,
    specifically designed to further reduce computational cost and optimize
    performance on resource-constrained mobile devices. It strikes a balance
    between accuracy and efficiency, with flexible width and resolution
    adjustments tailored for smaller models.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    32,455,856 weights, 326,138 biases -> 32,781,994 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    MobileNetBlock.InvRes()
    MobileNetBlock.InvRes_SE()
    ```
    Arguments
    ---------
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `dropout_rate` : float, default=0.2
        Dropout rate
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Howard, Andrew, et al. “Searching for MobileNetV3.” Proceedings
    of the IEEE/CVF International Conference on Computer Vision (ICCV),
    2019, pp. 1314-1324.

    """


class MobileNet_V3_L(mobile._MobileNet_V3_L):
    """
    MobileNet-V3-Large enhances its predecessors by integrating
    squeeze-and-excitation(SE) modules and the hard-swish activation,
    designed to boost performance while minimizing computational cost
    on mobile devices. It provides a balance of accuracy and efficiency,
    with flexible width and resolution adjustments optimized for larger,
    more powerful models.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    167,606,960 weights, 1,136,502 biases -> 168,743,462 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    MobileNetBlock.InvRes()
    MobileNetBlock.InvRes_SE()
    ```
    Arguments
    ---------
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `dropout_rate` : float, default=0.2
        Dropout rate
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Howard, Andrew, et al. “Searching for MobileNetV3.” Proceedings
    of the IEEE/CVF International Conference on Computer Vision (ICCV),
    2019, pp. 1314-1324.

    """


class SE_ResNet_50(resnet._SE_ResNet_50):
    """
    SE-ResNet is a deep neural network that extends the ResNet
    architecture by integrating Squeeze-and-Excitation blocks.
    These blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the
    network.

    ResNet-50 is the base network for this SE-augmented version.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    35,615,808 weights, 46,440 biases -> 35,662,248 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SE()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Hu, Jie, et al. “Squeeze-and-Excitation Networks.”
    Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 7132-7141.

    """


class SE_ResNet_152(resnet._SE_ResNet_152):
    """
    SE-ResNet is a deep neural network that extends the ResNet
    architecture by integrating Squeeze-and-Excitation blocks.
    These blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the
    network.

    ResNet-152 is the base network for this SE-augmented version.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    86,504,512 weights, 136,552 biases -> 86,641,064 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SE()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Hu, Jie, et al. “Squeeze-and-Excitation Networks.”
    Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 7132-7141.

    """


class SE_Inception_ResNet_V2(incep._SE_Inception_ResNet_V2):
    """
    SE-Inception-ResNet v2 is a deep neural network that extends the
    Inception-ResNet v2 architecture by integrating Squeeze-and-Excitation (SE)
    blocks. These SE blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the network by
    adaptively recalibrating the channel-wise feature responses.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 299, 299] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    58,794,080 weights, 80,762 biases -> 58,874,842 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    # These blocks are SE-augmented
    IncepResBlock.V2_TypeA(),
    IncepResBlock.V2_TypeB(),
    IncepResBlock.V2_TypeC(),
    IncepResBlock.V2_Redux(),

    IncepBlock.V4_Stem(),
    IncepBlock.V4_ReduxA()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `smoothing` : float, default=0.1
        Label smoothing factor
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Hu, Jie, et al. “Squeeze-and-Excitation Networks.”
    Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition (CVPR), 2018, pp. 7132-7141.

    """


class SE_DenseNet_121(dense._SE_DenseNet_121):
    """
    SE-DenseNet-121 is a deep neural network that extends the DenseNet-121
    architecture by integrating Squeeze-and-Excitation (SE) blocks.
    These SE blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the network
    by adaptively recalibrating the channel-wise feature responses.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    9,190,272 weights, 14,760 biases -> 9,205,032 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    # These blocks are SE-augmented
    DenseNetBlock.Composite(),
    DenseNetBlock.DenseUnit(),
    DenseNetBlock.Transition()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """


class SE_DenseNet_169(dense._SE_DenseNet_169):
    """
    SE-DenseNet-169 is a deep neural network that extends the DenseNet-169
    architecture by integrating Squeeze-and-Excitation (SE) blocks.
    These SE blocks enhance the network's ability to model channel-wise
    interdependencies, improving the representational power of the network
    by adaptively recalibrating the channel-wise feature responses.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    16,515,968 weights, 19,848 biases -> 16,535,816 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    # These blocks are SE-augmented
    DenseNetBlock.Composite(),
    DenseNetBlock.DenseUnit(),
    DenseNetBlock.Transition()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """


class SE_ResNeXt_50(resnext._SE_ResNeXt_50):
    """
    SE-ResNeXt-50 is a 50-layer deep neural network that builds upon
    ResNeXt by integrating Squeeze-and-Excitation (SE) blocks alongside
    the "cardinality" dimension, which refers to the number of independent
    paths within each residual block. This combination enhances
    representational power and efficiency by adaptively recalibrating
    channel-wise feature responses, allowing for greater flexibility in
    learning complex patterns, thereby improving performance in image
    recognition tasks while maintaining computational efficiency.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    37,135,680 weights, 53,992 biases -> 37,189,672 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SE()  # Override `expansion` to 2
    ```
    Arguments
    ---------
    `cardinality` : int, default=32
        The cardinality in terms of aggregated residual
        transformations.
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """


class SE_ResNeXt_101(resnext._SE_ResNeXt_101):
    """
    SE-ResNeXt-101 is a 101-layer deep neural network that builds upon
    ResNeXt by integrating Squeeze-and-Excitation (SE) blocks alongside
    the "cardinality" dimension, which refers to the number of independent
    paths within each residual block. This combination enhances
    representational power and efficiency by adaptively recalibrating
    channel-wise feature responses, allowing for greater flexibility in
    learning complex patterns, thereby improving performance in image
    recognition tasks while maintaining computational efficiency.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    65,197,376 weights, 110,568 biases -> 65,307,944 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SE()  # Override `expansion` to 2
    ```
    Arguments
    ---------
    `cardinality` : int, default=32
        The cardinality in terms of aggregated residual
        transformations.
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    """


class DenseNet_121(dense._DenseNet_121):
    """
    DenseNet-121 is a deep neural network architecture that connects each layer to
    every other layer in a feed-forward fashion. Unlike traditional architectures
    where layers are connected sequentially, DenseNet-121 establishes direct
    connections between any two layers with the same feature-map size, enabling
    the reuse of features.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    7,977,856 weights, 11,240 biases -> 7,989,096 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    DenseNetBlock.Composite(),
    DenseNetBlock.DenseUnit(),
    DenseNetBlock.Transition()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Huang, Gao, et al. "Densely Connected Convolutional Networks."
    Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition, 2017, pp. 4700-4708.

    """


class DenseNet_169(dense._DenseNet_169):
    """
    DenseNet-169 is a deep neural network architecture that connects each layer to
    every other layer in a feed-forward fashion. Unlike traditional architectures
    where layers are connected sequentially, DenseNet-169 establishes direct
    connections between any two layers with the same feature-map size, enabling
    the reuse of features.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    14,148,480 weights, 15,208 biases -> 14,163,688 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    DenseNetBlock.Composite(),
    DenseNetBlock.DenseUnit(),
    DenseNetBlock.Transition()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Huang, Gao, et al. "Densely Connected Convolutional Networks."
    Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition, 2017, pp. 4700-4708.

    """


class DenseNet_201(dense._DenseNet_201):
    """
    DenseNet-201 is a deep neural network architecture that connects each layer to
    every other layer in a feed-forward fashion. Unlike traditional architectures
    where layers are connected sequentially, DenseNet-201 establishes direct
    connections between any two layers with the same feature-map size, enabling
    the reuse of features.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    20,012,928 weights, 18,024 biases -> 20,030,952 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    DenseNetBlock.Composite(),
    DenseNetBlock.DenseUnit(),
    DenseNetBlock.Transition()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Huang, Gao, et al. "Densely Connected Convolutional Networks."
    Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition, 2017, pp. 4700-4708.

    """


class DenseNet_264(dense._DenseNet_264):
    """
    DenseNet-264 is a deep neural network architecture that connects each layer to
    every other layer in a feed-forward fashion. Unlike traditional architectures
    where layers are connected sequentially, DenseNet-264 establishes direct
    connections between any two layers with the same feature-map size, enabling
    the reuse of features.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    33,336,704 weights, 23,400 biases -> 33,360,104 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    DenseNetBlock.Composite(),
    DenseNetBlock.DenseUnit(),
    DenseNetBlock.Transition()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Huang, Gao, et al. "Densely Connected Convolutional Networks."
    Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition, 2017, pp. 4700-4708.

    """


class EfficientNet_B0(efficient._EfficientNet_B0):
    """
    EfficientNet-B0 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    4,803,040 weights, 24,268 biases -> 4,827,308 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_B1(efficient._EfficientNet_B1):
    """
    EfficientNet-B1 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 240, 240] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    6,544,500 weights, 32,568 biases -> 6,577,068 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_B2(efficient._EfficientNet_B2):
    """
    EfficientNet-B2 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 260, 260] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    8,503,007 weights, 40,160 biases -> 8,543,167 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_B3(efficient._EfficientNet_B3):
    """
    EfficientNet-B3 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 300, 300] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    13,657,980 weights, 57,390 biases -> 13,715,370 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_B4(efficient._EfficientNet_B4):
    """
    EfficientNet-B4 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 380, 380] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    17,877,155 weights, 72,278 biases -> 17,949,433 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_B5(efficient._EfficientNet_B5):
    """
    EfficientNet-B5 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 456, 456] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    24,674,011 weights, 94,261 biases -> 24,768,272 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_B6(efficient._EfficientNet_B6):
    """
    EfficientNet-B6 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 528, 528] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    38,260,230 weights, 132,704 biases -> 38,392,934 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_B7(efficient._EfficientNet_B7):
    """
    EfficientNet-B7 is the baseline model in the EfficientNet family,
    designed using compound scaling to optimize width, depth, and resolution.
    It builds on MobileNetV3, utilizing inverted residual blocks and
    Squeeze-and-Excitation (SE) blocks to improve efficiency and performance
    on tasks like ImageNet classification.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 600, 600] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    56,528,906 weights, 178,066 biases -> 56,706,972 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks." International Conference on Machine
    Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    """


class EfficientNet_V2_S(efficient._EfficientNet_V2_S):
    """
    EfficientNet-V2-S is a smaller variant in the EfficientNet-V2 family,
    designed with improved training speed and better parameter efficiency
    compared to its predecessor. It introduces advancements like Fused-MBConv
    blocks, which combine depthwise convolutions and regular convolutions for
    faster computation, and further optimizes the scaling of width, depth, and
    resolution.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 384, 384] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    18,414,552 weights, 86,116 biases -> 18,500,668 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv(),
    EfficientBlock.FusedMBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `dropout_rate` : float, default=0.1
        Initial dropout rate
    `progressive_learning` : bool, default=True
        Whether to perform a progressive learning mentioned in the paper
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc Le. “EfficientNetV2: Smaller Models and
    Faster Training.” Proceedings of the 38th International Conference on
    Machine Learning (ICML), vol. 139, 2021, pp. 10096-10106.

    """


class EfficientNet_V2_M(efficient._EfficientNet_V2_M):
    """
    EfficientNet-V2-M is a medium-sized variant in the EfficientNet-V2 family,
    designed with improved training speed and better parameter efficiency
    compared to its predecessor. It introduces advancements like Fused-MBConv
    blocks, which combine depthwise convolutions and regular convolutions for
    faster computation, and further optimizes the scaling of width, depth, and
    resolution.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 480, 480] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    46,012,920 weights, 162,264 biases -> 46,175,184 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv(),
    EfficientBlock.FusedMBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `dropout_rate` : float, default=0.1
        Initial dropout rate
    `progressive_learning` : bool, default=True
        Whether to perform a progressive learning mentioned in the paper
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc Le. “EfficientNetV2: Smaller Models and
    Faster Training.” Proceedings of the 38th International Conference on
    Machine Learning (ICML), vol. 139, 2021, pp. 10096-10106.

    """


class EfficientNet_V2_L(efficient._EfficientNet_V2_L):
    """
    EfficientNet-V2-L is a larger variant in the EfficientNet-V2 family,
    designed with improved training speed and better parameter efficiency
    compared to its predecessor. It introduces advancements like Fused-MBConv
    blocks, which combine depthwise convolutions and regular convolutions for
    faster computation, and further optimizes the scaling of width, depth, and
    resolution.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 480, 480] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    104,084,896 weights, 303,032 biases -> 104,387,928 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv(),
    EfficientBlock.FusedMBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `dropout_rate` : float, default=0.1
        Initial dropout rate
    `progressive_learning` : bool, default=True
        Whether to perform a progressive learning mentioned in the paper
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc Le. “EfficientNetV2: Smaller Models and
    Faster Training.” Proceedings of the 38th International Conference on
    Machine Learning (ICML), vol. 139, 2021, pp. 10096-10106.

    """


class EfficientNet_V2_XL(efficient._EfficientNet_V2_XL):
    """
    EfficientNet-V2-XL is the largest model in the EfficientNet-V2 family,
    designed specifically for large-scale datasets like ImageNet-21k.
    It employs advanced compound scaling and Fused-MBConv blocks to optimize
    both training speed and accuracy. EfficientNetV2-XL is particularly suited
    for high-performance applications requiring extensive computational power,
    offering state-of-the-art results on large classification tasks.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 480, 480] -> Matrix[-1, 21,843]
    ```
    Parameter Size:
    ```
    201,762,976 weights, 450,227 biases -> 202,213,203 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    EfficientBlock.MBConv(),
    EfficientBlock.FusedMBConv()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.Swish
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=21843
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `dropout_rate` : float, default=0.1
        Initial dropout rate
    `progressive_learning` : bool, default=True
        Whether to perform a progressive learning mentioned in the paper
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Tan, Mingxing, and Quoc Le. “EfficientNetV2: Smaller Models and
    Faster Training.” Proceedings of the 38th International Conference on
    Machine Learning (ICML), vol. 139, 2021, pp. 10096-10106.

    """


class ResNeXt_50(resnext._ResNeXt_50):
    """
    ResNeXt-50 is a 50-layer deep neural network that builds upon ResNet
    by introducing a "cardinality" dimension, which refers to the number of
    independent paths within each residual block. This design improves
    representational power and efficiency, allowing for more flexibility in
    learning complex patterns, enhancing performance in image recognition
    tasks while maintaining computational efficiency.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    25,027,904 weights, 35,112 biases -> 25,063,016 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()  # Override `expansion` to 2
    ```
    Arguments
    ---------
    `cardinality` : int, default=32
        The cardinality in terms of aggregated residual
        transformations.
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Xie, Saining, et al. "Aggregated Residual Transformations
    for Deep Neural Networks." Proceedings of the IEEE Conference
    on Computer Vision and Pattern Recognition (CVPR), 2017,
    pp. 1492-1500.

    """


class ResNeXt_101(resnext._ResNeXt_101):
    """
    ResNeXt-101 is a 101-layer deep neural network that builds upon ResNet
    by introducing a "cardinality" dimension, which refers to the number of
    independent paths within each residual block. This design improves
    representational power and efficiency, allowing for more flexibility in
    learning complex patterns, enhancing performance in image recognition
    tasks while maintaining computational efficiency.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```txt
    44,176,704 weights, 69,928 biases -> 44,246,632 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck()  # Override `expansion` to 2
    ```
    Arguments
    ---------
    `cardinality` : int, default=32
        The cardinality in terms of aggregated residual
        transformations.
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    1. Xie, Saining, et al. "Aggregated Residual Transformations
    for Deep Neural Networks." Proceedings of the IEEE Conference
    on Computer Vision and Pattern Recognition (CVPR), 2017,
    pp. 1492-1500.

    """


class SK_ResNet_50(resnet._SK_ResNet_50):
    """
    SK-ResNet is a deep neural network that extends the ResNet architecture
    by incorporating Selective Kernel (SK) blocks. These blocks dynamically
    adjust the receptive field by adaptively selecting kernels of different
    sizes, enhancing the network's ability to capture multi-scale features
    and improving its representational power.

    ResNet-50 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    57,236,160 weights, 39,124 biases -> 57,275,284 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SK()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Li, Xiang, et al. "Selective Kernel Networks." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR), 2019, pp. 510-519.

    """


class SK_ResNet_101(resnet._SK_ResNet_101):
    """
    SK-ResNet is a deep neural network that extends the ResNet architecture
    by incorporating Selective Kernel (SK) blocks. These blocks dynamically
    adjust the receptive field by adaptively selecting kernels of different
    sizes, enhancing the network's ability to capture multi-scale features
    and improving its representational power.

    ResNet-101 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    104,298,688 weights, 78,564 biases -> 104,377,252 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SK()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Li, Xiang, et al. "Selective Kernel Networks." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR), 2019, pp. 510-519.

    """


class SK_ResNeXt_50(resnext._SK_ResNeXt_50):
    """
    SK-ResNeXt is a deep neural network that extends the ResNeXt architecture
    by integrating Selective Kernel (SK) blocks. These blocks dynamically
    adjust the receptive field by adaptively selecting kernels of varying
    sizes, enhancing the model's ability to capture multi-scale features.
    This integration improves the representational power of the network,
    allowing for more efficient and effective feature learning.

    ResNeXt-50 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    29,915,712 weights, 58,240 biases -> 29,973,952 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SK()  # Override `expantion` to 2
    ```
    Arguments
    ---------
    `cardinality` : int, default=32
        The cardinality in terms of aggregated residual transformations.
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Li, Xiang, et al. "Selective Kernel Networks." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR), 2019, pp. 510-519.

    """


class SK_ResNeXt_101(resnext._SK_ResNeXt_101):
    """
    SK-ResNeXt is a deep neural network that extends the ResNeXt architecture
    by integrating Selective Kernel (SK) blocks. These blocks dynamically
    adjust the receptive field by adaptively selecting kernels of varying
    sizes, enhancing the model's ability to capture multi-scale features.
    This integration improves the representational power of the network,
    allowing for more efficient and effective feature learning.

    ResNeXt-101 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    53,399,104 weights, 119,712 biases -> 53,518,816 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNetBlock.Bottleneck_SK()  # Override `expantion` to 2
    ```
    Arguments
    ---------
    `cardinality` : int, default=32
        The cardinality in terms of aggregated residual transformations.
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Li, Xiang, et al. "Selective Kernel Networks." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR), 2019, pp. 510-519.

    """


class ResNeSt_50(resnest._ResNeSt_50):
    """
    ResNeSt (Residual Networks with Squeeze-and-Excitation Networks and
    Split-Attention) is a variant of ResNet that improves performance by
    using split-attention blocks to enhance feature aggregation and
    channel-wise attention. It achieves state-of-the-art results in image
    classification and object detection tasks.

    ResNet-50 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    26,535,136 weights, 39,944 biases -> 26,575,080 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNeStBlock()
    ```
    Arguments
    ---------
    `radix` : int, default=2
        The number of 'splits' of Split-Attention module
    `cardinality` : int, default=1
        The cardinality referring to the number of 'groups'
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." arXiv
    preprint arXiv:2004.08955 (2020).

    """


class ResNeSt_101(resnest._ResNeSt_101):
    """
    ResNeSt (Residual Networks with Squeeze-and-Excitation Networks and
    Split-Attention) is a variant of ResNet that improves performance by
    using split-attention blocks to enhance feature aggregation and
    channel-wise attention. It achieves state-of-the-art results in image
    classification and object detection tasks.

    ResNet-101 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    46,371,552 weights, 80,200 biases -> 46,451,752 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNeStBlock()
    ```
    Arguments
    ---------
    `radix` : int, default=2
        The number of 'splits' of Split-Attention module
    `cardinality` : int, default=1
        The cardinality referring to the number of 'groups'
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." arXiv
    preprint arXiv:2004.08955 (2020).

    """


class ResNeSt_200(resnest._ResNeSt_200):
    """
    ResNeSt (Residual Networks with Squeeze-and-Excitation Networks and
    Split-Attention) is a variant of ResNet that improves performance by
    using split-attention blocks to enhance feature aggregation and
    channel-wise attention. It achieves state-of-the-art results in image
    classification and object detection tasks.

    ResNet-200 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    67,392,736 weights, 134,664 biases -> 67,527,400 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNeStBlock()
    ```
    Arguments
    ---------
    `radix` : int, default=2
        The number of 'splits' of Split-Attention module
    `cardinality` : int, default=1
        The cardinality referring to the number of 'groups'
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." arXiv
    preprint arXiv:2004.08955 (2020).

    """


class ResNeSt_269(resnest._ResNeSt_269):
    """
    ResNeSt (Residual Networks with Squeeze-and-Excitation Networks and
    Split-Attention) is a variant of ResNet that improves performance by
    using split-attention blocks to enhance feature aggregation and
    channel-wise attention. It achieves state-of-the-art results in image
    classification and object detection tasks.

    ResNet-269 is the base model for this variation.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    106,451,680 weights, 193,864 biases -> 106,645,544 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ResNeStBlock()
    ```
    Arguments
    ---------
    `radix` : int, default=2
        The number of 'splits' of Split-Attention module
    `cardinality` : int, default=1
        The cardinality referring to the number of 'groups'
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." arXiv
    preprint arXiv:2004.08955 (2020).

    """


class ConvNeXt_T(convnext._ConvNeXt_T):
    """
    ConvNeXt-T (ConvNeXt-Tiny) is a lightweight variant of the ConvNeXt
    architecture, designed to be more efficient while maintaining strong
    performance on image classification tasks. It utilizes modernized
    convolutional blocks with depthwise convolutions, layer normalization,
    and residual connections to improve feature extraction.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    28,524,000 weights, 42,184 biases -> 28,566,184 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvNeXtBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.GELU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    2022, pp. 11976-11986.

    """


class ConvNeXt_S(convnext._ConvNeXt_S):
    """
    ConvNeXt-S, (ConvNeXt-Small) is a mid-sized variant of the ConvNeXt
    architecture, offering more capacity and depth than ConvNeXt-Tiny while
    still being relatively efficient. It uses modernized convolutional blocks,
    including depthwise convolutions, layer normalization, and residual
    connections, to enhance feature learning.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    50,096,352 weights, 83,656 biases -> 50,180,008 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvNeXtBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.GELU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    2022, pp. 11976-11986.

    """


class ConvNeXt_B(convnext._ConvNeXt_B):
    """
    ConvNeXt-B, (ConvNeXt-Base) is a standard-sized version of the ConvNeXt
    architecture, designed for achieving high accuracy on image classification
    tasks. It incorporates deeper layers and more parameters compared to smaller
    variants, using the same advanced convolutional blocks and residual
    connections.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    88,422,016 weights, 111,208 biases -> 88,533,224 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvNeXtBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.GELU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    2022, pp. 11976-11986.

    """


class ConvNeXt_L(convnext._ConvNeXt_L):
    """
    ConvNeXt-L, (ConvNeXt-Large) is a larger variant of the ConvNeXt architecture,
    designed to maximize performance on large-scale datasets. It uses more layers
    and parameters than ConvNeXt-B, leveraging depthwise convolutions and residual
    connections to enhance feature extraction.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    197,513,664 weights, 166,312 biases -> 197,679,976 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvNeXtBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.GELU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    2022, pp. 11976-11986.

    """


class ConvNeXt_XL(convnext._ConvNeXt_XL):
    """
    ConvNeXt-XL, (ConvNeXt-Extra Large) is the largest variant in the ConvNeXt
    family, offering the highest capacity for complex image classification tasks.
    It features a significantly increased number of layers and parameters,
    utilizing the same modernized convolutional design to capture detailed
    features.

    Specs
    -----
    Input/Output Shapes:
    ```py
    Tensor[-1, 3, 224, 224] -> Matrix[-1, 1000]
    ```
    Parameter Size:
    ```
    349,859,072 weights, 221,416 biases -> 350,080,488 params
    ```
    Components
    ----------
    Blocks Used:
    ```py
    ConvNeXtBlock()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.GELU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=1000
        Number of output features
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0001
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    2022, pp. 11976-11986.

    """
