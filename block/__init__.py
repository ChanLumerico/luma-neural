"""
`neural.block`
--------------
Neural blocks are modular units in computational systems, each performing 
specific operations like data processing or transformation. They are 
composed of multiple layers or steps and can be combined to build complex 
structures. These blocks simplify the design and functionality of advanced 
systems by breaking down tasks into manageable components.

"""

from dataclasses import dataclass
from typing import Literal, Tuple

from luma.core.super import Optimizer
from luma.interface.typing import ClassType
from luma.interface.util import InitUtil

from luma.neural.block import (
    convnext,
    dense,
    efficient,
    incep,
    incep_res,
    mobile,
    resnest,
    resnet,
    se,
    sk,
    standard,
    xception,
)


__all__ = (
    "ConvBlock1D",
    "ConvBlock2D",
    "ConvBlock3D",
    "SeparableConv1D",
    "SeparableConv2D",
    "SeparableConv3D",
    "DenseBlock",
    "SEBlock1D",
    "SEBlock2D",
    "SEBlock3D",
    "SKBlock1D",
    "SKBlock2D",
    "SKBlock3D",
    "IncepBlock",
    "IncepResBlock",
    "ResNetBlock",
    "XceptionBlock",
    "MobileNetBlock",
    "DenseNetBlock",
    "EfficientBlock",
    "ResNeStBlock",
    "ConvNeXtBlock",
)


@dataclass
class ConvBlockArgs:
    filter_size: Tuple[int, ...] | int
    activation: callable
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    padding: Tuple[int, ...] | int | Literal["same", "valid"] = "same"
    stride: int = 1
    lambda_: float = 0.0
    do_batch_norm: bool = True
    momentum: float = 0.9
    do_pooling: bool = True
    pool_filter_size: int = 2
    pool_stride: int = 2
    pool_mode: Literal["max", "avg"] = "max"
    random_state: int | None = None


class ConvBlock1D(standard._ConvBlock1D):
    """
    Convolutional block for 1-dimensional data.

    A convolutional block in a neural network typically consists of a
    convolutional layer followed by a nonlinear activation function,
    and a pooling layer to reduce spatial dimensions.
    This structure extracts and transforms features from input data,
    applying filters to capture spatial hierarchies and patterns.
    The pooling layer then reduces the feature dimensionality, helping to
    decrease computational cost and overfitting.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `activation` : callable
        Type of activation function
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Filter size for pooling
    `pool_stride` : int, default=2
        Step size for pooling process
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """


class ConvBlock2D(standard._ConvBlock2D):
    """
    Convolutional block for 2-dimensional data.

    A convolutional block in a neural network typically consists of a
    convolutional layer followed by a nonlinear activation function,
    and a pooling layer to reduce spatial dimensions.
    This structure extracts and transforms features from input data,
    applying filters to capture spatial hierarchies and patterns.
    The pooling layer then reduces the feature dimensionality, helping to
    decrease computational cost and overfitting.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `activation` : callable
        Type of activation function
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Filter size for pooling
    `pool_stride` : int, default=2
        Step size for pooling process
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """


class ConvBlock3D(standard._ConvBlock3D):
    """
    Convolutional block for 3-dimensional data.

    A convolutional block in a neural network typically consists of a
    convolutional layer followed by a nonlinear activation function,
    and a pooling layer to reduce spatial dimensions.
    This structure extracts and transforms features from input data,
    applying filters to capture spatial hierarchies and patterns.
    The pooling layer then reduces the feature dimensionality, helping to
    decrease computational cost and overfitting.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `activation` : callable
        Type of activation function
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Filter size for pooling
    `pool_stride` : int, default=2
        Step size for pooling process
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """


class SeparableConv1D(standard._SeparableConv1D):
    """
    Depthwise Seperable Convolutional(DSC) block for
    1-dimensional data.

    Depthwise separable convolution(DSC) splits convolution into
    depthwise (per-channel) and pointwise (1x1) steps, reducing
    computation and parameters while preserving performance,
    often used in efficient models like MobileNet.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of 3D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, width)
        ```
    """


class SeparableConv2D(standard._SeparableConv2D):
    """
    Depthwise Seperable Convolutional(DSC) block for
    2-dimensional data.

    Depthwise separable convolution(DSC) splits convolution into
    depthwise (per-channel) and pointwise (1x1) steps, reducing
    computation and parameters while preserving performance,
    often used in efficient models like MobileNet.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of 4D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, height, width)
        ```
    """


class SeparableConv3D(standard._SeparableConv3D):
    """
    Depthwise Seperable Convolutional(DSC) block for
    3-dimensional data.

    Depthwise separable convolution(DSC) splits convolution into
    depthwise (per-channel) and pointwise (1x1) steps, reducing
    computation and parameters while preserving performance,
    often used in efficient models like MobileNet.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_size`: tuple of int or int
        Size of each filter
    `padding` : tuple of int or int or {"same", "valid"}, default="same"
        Padding method
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight updating
    `initializer` : InitStr, default=None
        Type of weight initializer
    `stride` : int, default=1
        Step size for filters during convolution
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=False
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization

    Notes
    -----
    - The input `X` must have the form of 5D-array(`Tensor`).

        ```py
        X.shape = (batch_size, channels, depth, height, width)
        ```
    """


@dataclass
class DenseBlockArgs:
    activation: callable
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    lambda_: float = 0.0
    do_batch_norm: float = True
    momentum: float = 0.9
    do_dropout: bool = True
    dropout_rate: float = 0.5
    random_state: int | None = None


class DenseBlock(standard._DenseBlock):
    """
    A typical dense block in a neural network configuration often
    includes a series of fully connected (dense) layers. Each layer
    within the block connects every input neuron to every output
    neuron through learned weights. Activation functions, such as ReLU,
    are applied after each dense layer to introduce non-linear processing,
    enhancing the network's ability to learn complex patterns.
    Optionally, dropout or other regularization techniques may be
    included to reduce overfitting by randomly deactivating a portion
    of the neurons during training.

    Parameters
    ----------
    `in_features` : int
        Number of input features
    `out_features` : int
        Number of output features
    `activation` : callable
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_dropout` : bool, default=True
        Whethter to perform dropout
    `dropout_rate` : float, default=0.5
        Dropout rate

    """


class SEBlock1D(se._SEBlock1D):
    """
    Squeeze-and-Excitation(SE) block for 1-dimensional data.

    The SE-Block enhances the representational
    power of a network by recalibrating channel-wise feature responses. It
    first squeezes the spatial dimensions using global average pooling, then
    excites the channels with learned weights through fully connected layers
    and an activation function. This selectively emphasizes important channels
    while suppressing less relevant ones.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `reduction`: int, default=4
        Reducing factor of the 'Squeeze' phase.
    `activation` : callable, default=Activation.HardSwish
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `keep_shape` : bool, default=True
        Whether to maintain the original shape of the input;
        Transforms 3D-Tensor to 2D-Matrix if set to False.

    """


class SEBlock2D(se._SEBlock2D):
    """
    Squeeze-and-Excitation(SE) block for 2-dimensional data.

    The SE-Block enhances the representational
    power of a network by recalibrating channel-wise feature responses. It
    first squeezes the spatial dimensions using global average pooling, then
    excites the channels with learned weights through fully connected layers
    and an activation function. This selectively emphasizes important channels
    while suppressing less relevant ones.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `reduction`: int, default=4
        Reducing factor of the 'Squeeze' phase.
    `activation` : callable, default=Activation.HardSwish
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `keep_shape` : bool, default=True
        Whether to maintain the original shape of the input;
        Transforms 4D-Tensor to 2D-Matrix if set to False.

    """


class SEBlock3D(se._SEBlock3D):
    """
    Squeeze-and-Excitation(SE) block for 3-dimensional data.

    The SE-Block enhances the representational
    power of a network by recalibrating channel-wise feature responses. It
    first squeezes the spatial dimensions using global average pooling, then
    excites the channels with learned weights through fully connected layers
    and an activation function. This selectively emphasizes important channels
    while suppressing less relevant ones.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `reduction`: int, default=4
        Reducing factor of the 'Squeeze' phase.
    `activation` : callable, default=Activation.HardSwish
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength
    `keep_shape` : bool, default=True
        Whether to maintain the original shape of the input;
        Transforms 5D-Tensor to 2D-Matrix if set to False.

    """


class SKBlock1D(sk._SKBlock1D):
    """
    Selective Kernel(SK) block for 1-dimensional data.

    The SK Block dynamically selects between different convolutional kernel
    sizes to capture multi-scale features. It processes input through multiple
    kernels, fuses the results, and uses attention to adaptively focus on the
    most relevant features.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_sizes`: list of int, default=[3, 5]
        List of filter sizes for each branches
    `reduction` : int, default=16
        Reducing factor of FC layer in fusing phase
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength

    """


class SKBlock2D(sk._SKBlock2D):
    """
    Selective Kernel(SK) block for 2-dimensional data.

    The SK Block dynamically selects between different convolutional kernel
    sizes to capture multi-scale features. It processes input through multiple
    kernels, fuses the results, and uses attention to adaptively focus on the
    most relevant features.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_sizes`: list of int, default=[3, 5]
        List of filter sizes for each branches
    `reduction` : int, default=16
        Reducing factor of FC layer in fusing phase
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength

    """


class SKBlock3D(sk._SKBlock3D):
    """
    Selective Kernel(SK) block for 3-dimensional data.

    The SK Block dynamically selects between different convolutional kernel
    sizes to capture multi-scale features. It processes input through multiple
    kernels, fuses the results, and uses attention to adaptively focus on the
    most relevant features.

    Parameters
    ----------
    `in_channels` : int
        Number of input channels
    `out_channels` : int
        Number of output channels
    `filter_sizes`: list of int, default=[3, 5]
        List of filter sizes for each branches
    `reduction` : int, default=16
        Reducing factor of FC layer in fusing phase
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `optimizer` : Optimizer, optional, default=None
        Type of optimizer for weight update
    `initializer` : InitStr, default=None
        Type of weight initializer
    `lambda_` : float, default=0.0
        L2 regularization strength

    """


@dataclass
class BaseBlockArgs:
    activation: callable
    optimizer: Optimizer | None = None
    initializer: InitUtil.InitStr = None
    lambda_: float = 0.0
    do_batch_norm: float = True
    momentum: float = 0.9
    random_state: int | None = None


@ClassType.non_instantiable()
class IncepBlock:
    """
    Container class for various Inception blocks.

    References
    ----------
    `Inception V1, V2` :
        [1] Szegedy, Christian, et al. “Going Deeper with Convolutions.”
        Proceedings of the IEEE Conference on Computer Vision and
        Pattern Recognition (CVPR), 2015, pp. 1-9,
        arxiv.org/abs/1409.4842.

    `Inception V4` :
        [2] Szegedy, Christian, et al. “Inception-v4, Inception-ResNet
        and the Impact of Residual Connections on Learning.”
        Proceedings of the Thirty-First AAAI Conference on
        Artificial Intelligence (AAAI), 2017, pp. 4278-4284,
        arxiv.org/abs/1602.07261.

    """

    class V1(incep._Incep_V1_Default):
        """
        Inception block for Inception V1 network, a.k.a. GoogLeNet.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeA(incep._Incep_V2_TypeA):
        """
        Inception block type-A for Inception V2 network.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeB(incep._Incep_V2_TypeB):
        """
        Inception block type-B for Inception V2 network.

        Refer to the figures shown in the original paper[1].
        """

    class V2_TypeC(incep._Incep_V2_TypeC):
        """
        Inception block type-C for Inception V2 network.

        Refer to the figures shown in the original paper[1].

        """

    class V2_Redux(incep._Incep_V2_Redux):
        """
        Inception block for grid reduction for Inception V2 network.

        Refer to the figures shown in the original paper[1].

        """

    class V4_Stem(incep._Incep_V4_Stem):
        """
        Inception block used in Inception V4 network stem part.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 3, 299, 299]
            Output: Tensor[-1, 384, 35, 35]
            ```
        """

    class V4_TypeA(incep._Incep_V4_TypeA):
        """
        Inception block type A used in Inception V4 network

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 384, 35, 35]
            Output: Tensor[-1, 384, 35, 35]
            ```
        """

    class V4_TypeB(incep._Incep_V4_TypeB):
        """
        Inception block type B used in Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1024, 17, 17]
            Output: Tensor[-1, 1024, 17, 17]
            ```
        """

    class V4_TypeC(incep._Incep_V4_TypeC):
        """
        Inception block type C used in Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1536, 8, 8]
            Output: Tensor[-1, 1536, 8, 8]
            ```
        """

    class V4_ReduxA(incep._Incep_V4_ReduxA):
        """
        Inception block type A for grid reduction used in
        Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 384, 35, 35]
            Output: Tensor[-1, 1024, 17, 17]
            ```
        """

    class V4_ReduxB(incep._Incep_V4_ReduxB):
        """
        Inception block type B for grid reduction used in
        Inception V4 network.

        Refer to the figures shown in the original paper[2].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1024, 17, 17]
            Output: Tensor[-1, 1536, 8, 8]
            ```
        """


@ClassType.non_instantiable()
class IncepResBlock:
    """
    Container class for various Inception-ResNet blocks.

    References
    ----------
    `Inception-ResNet V1, V2` :
        [1] Szegedy, Christian, et al. “Inception-v4, Inception-ResNet
        and the Impact of Residual Connections on Learning.”
        Proceedings of the Thirty-First AAAI Conference on
        Artificial Intelligence (AAAI), 2017, pp. 4278-4284,
        arxiv.org/abs/1602.07261.

    """

    class V1_Stem(incep_res._IncepRes_V1_Stem):
        """
        Inception block used in Inception-ResNet V1 network
        stem part.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 3, 299, 299]
            Output: Tensor[-1, 256, 35, 35]
            ```
        """

    class V1_TypeA(incep_res._IncepRes_V1_TypeA):
        """
        Inception block type A used in Inception-ResNet V1
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 256, 35, 35]
            Output: Tensor[-1, 256, 35, 35]
            ```
        """

    class V1_TypeB(incep_res._IncepRes_V1_TypeB):
        """
        Inception block type B used in Inception-ResNet V1
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 892, 17, 17]
            Output: Tensor[-1, 892, 17, 17]
            ```
        """

    class V1_TypeC(incep_res._IncepRes_V1_TypeC):
        """
        Inception block type C used in Inception-ResNet V1
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1792, 8, 8]
            Output: Tensor[-1, 1792, 8, 8]
            ```
        """

    class V1_Redux(incep_res._IncepRes_V1_Redux):
        """
        Inception block type B for grid reduction used in
        Inception-ResNet V1 network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 896, 17, 17]
            Output: Tensor[-1, 1792, 8, 8]
            ```
        """

    class V2_TypeA(incep_res._IncepRes_V2_TypeA):
        """
        Inception block type A used in Inception-ResNet V2
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 384, 35, 35]
            Output: Tensor[-1, 384, 35, 35]
            ```
        """

    class V2_TypeB(incep_res._IncepRes_V2_TypeB):
        """
        Inception block type B used in Inception-ResNet V2
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1280, 17, 17]
            Output: Tensor[-1, 1280, 17, 17]
            ```
        """

    class V2_TypeC(incep_res._IncepRes_V2_TypeC):
        """
        Inception block type C used in Inception-ResNet V2
        network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 2272, 8, 8]
            Output: Tensor[-1, 2272, 8, 8]
            ```
        """

    class V2_Redux(incep_res._IncepRes_V2_Redux):
        """
        Inception block type B for grid reduction used in
        Inception-ResNet V2 network.

        Refer to the figures shown in the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 1289, 17, 17]
            Output: Tensor[-1, 2272, 8, 8]
            ```
        """


@ClassType.non_instantiable()
class ResNetBlock:
    """
    Container class for building components of ResNet.

    References
    ----------
    `ResNet-(18, 34, 50, 101, 152)` :
        [1] He, Kaiming, et al. “Deep Residual Learning for Image
        Recognition.” Proceedings of the IEEE Conference on Computer
        Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.

    `ResNet-(200, 269, 1001)` :
        [2] He, Kaiming, et al. “Identity Mappings in Deep Residual
        Networks.” European Conference on Computer Vision (ECCV),
        2016, pp. 630-645.

    `ResNeXt-(50, 101)` :
        [3] Xie, Saining, et al. "Aggregated Residual Transformations
        for Deep Neural Networks." Proceedings of the IEEE Conference
        on Computer Vision and Pattern Recognition (CVPR),
        2017, pp. 1492-1500.

    """

    class Basic(resnet._Basic):
        """
        Basic convolution block used in `ResNet-18` and `ResNet-34`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional, default=None
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling

        See [1] also for additional information.
        """

    class Bottleneck(resnet._Bottleneck):
        """
        Bottleneck block used in `ResNet-(50, 101, 152)`.

        If `groups` is greater than 1, it is served for
        `ResNeXt-50` and `ResNeXt-101`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional, default=None
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling
        `groups` : int, default=1
            Number of convolutional groups

        See [1] and [3] also for additional information.
        """

    class PreActBottleneck(resnet._PreActBottleneck):
        """
        Bottleneck block with pre-activation used in
        `ResNet-(200, 269, 1001)`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional, default=None
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling

        See [2] also for additional information.
        """

    class Bottleneck_SE(resnet._Bottleneck_SE):
        """
        Bottleneck block with squeeze-and-excitation(SE)
        used in `SE-ResNet`.

        If `groups` is greater than 1, it is served for
        `SE_ResNeXt`.

        Parameters
        ----------
        `se_reduction` : float, default=4
            Reduction factor for SE block
        `downsampling` : LayerLike, optional, default=None
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling
        `groups` : int, default=1
            Number of convolutional groups

        """

    class Bottleneck_SK(resnet._Bottleneck_SK):
        """
        Bottleneck block with selective kernel(SK)
        used in `SK-ResNet`.

        If `groups` is greater than 1, it is served for
        `SK_ResNeXt`.

        Parameters
        ----------
        `downsampling` : LayerLike, optional, default=None
            An additional layer to the input signal which reduces
            its grid size to perform a downsampling
        `filter_sizes` : list of int, default=[3, 5]
            List of filter sizes for each branches
        `reduction` : int, default=16
            Reducing factor of FC layer in fusing phase of SK block
        `groups` : int, default=1
            Number of convolutional groups

        """


@ClassType.non_instantiable()
class XceptionBlock:
    """
    Container class for building components of XceptionNet.

    References
    ----------
    `XceptionNet` :
        [1] Chollet, François. “Xception: Deep Learning with Depthwise
        Separable Convolutions.” Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2017,
        pp. 1251-1258.

    """

    class Entry(xception._Entry):
        """
        An entry flow of Xception network mentioned in Fig. 5
        of the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 3, 299, 299]
            Output: Tensor[-1, 728, 19, 19]
            ```
        """

    class Middle(xception._Middle):
        """
        A middle flow of Xception network mentioned in Fig. 5
        of the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 728, 19, 19]
            Output: Tensor[-1, 728, 19, 19]
            ```
        """

    class Exit(xception._Exit):
        """
        An exit flow of Xception network mentioned in Fig. 5
        of the original paper[1].

        Notes
        -----
        - This block has fixed shape of input and ouput tensors.

            ```py
            Input: Tensor[-1, 728, 19, 19]
            Output: Tensor[-1, 1024, 9, 9]
            ```
        """


@ClassType.non_instantiable()
class MobileNetBlock:
    """
    Container class for building components of MobileNet series.

    References
    ----------
    `MobileNet V2` :
        [1] Howard, Andrew G., et al. “MobileNets: Efficient
        Convolutional Neural Networks for Mobile Vision Applications.”
        arXiv, 17 Apr. 2017, arxiv.org/abs/1704.04861.

    `MobileNet V3` :
        [2] Howard, Andrew, et al. "Searching for MobileNetV3." Proceedings
        of the IEEE/CVF International Conference on Computer Vision, 2019,
        doi:10.1109/ICCV.2019.00140.

    """

    class InvRes(mobile._InvRes):
        """
        Inverted Residual Block with depth-wise and point-wise
        convolutions used in MobileNet V2.

        Refer to the figures shown in the original paper[1].
        """

    class InvRes_SE(mobile._InvRes_SE):
        """
        Inverted Residual Block with depth-wise and point-wise
        convolutions and SE-Block attached used in MobileNet V3.

        Parameters
        ----------
        `se_reduction` : float, default=4
            Reduction factor for SE block

        Refer to the figures shown in the original paper[2].
        """


@ClassType.non_instantiable()
class DenseNetBlock:
    """
    Container class for building components of DenseNet series.

    References
    ----------
    `DenseNet-(121, 169, 201, 264)` :
        [1] Huang, Gao, et al. "Densely Connected Convolutional Networks."
        Proceedings of the IEEE Conference on Computer Vision and Pattern
        Recognition, 2017, pp. 4700-4708.

    """

    class Composite(dense._Composite):
        """
        Composite function(H) used in DenseNet architecture.

        Parameters
        ----------
        `growth_rate` : int
            Growth rate of the channels
        `bn_size` : int, default=4
            Bottleneck size

        Refer to the figures shown in the original paper[1].
        """

    class DenseUnit(dense._DenseUnit):
        """
        Dense layer used as a unit component of DenseNet architecture.

        Parameters
        ----------
        `n_layers` : int
            Number of densely connected layers
        `growth_rate` : int
            Growth rate of the channels
        `bn_size` : int, default=4
            Bottleneck size

        Refer to the figures shown in the original paper[1].
        """

    class Transition(dense._Transition):
        """
        Transition layer used in DenseNet architecture.

        Parameters
        ----------
        `compression` : float, default=1.0
            Compression rate of the output channels

        Refer to the figures shown in the original paper[1].
        """


@ClassType.non_instantiable()
class EfficientBlock:
    """
    Container class for building components of EfficientNet series.

    References
    ----------
    `EfficientNet-(B0~B7)` :
        [1] Tan, Mingxing, and Quoc V. Le. "EfficientNet: Rethinking
        Model Scaling for Convolutional Neural Networks." International
        Conference on Machine Learning, 2020, pp. 6105-6114. arXiv:1905.11946.

    `EfficientNet-v2` :
        [2] Tan, Mingxing, and Quoc Le. “EfficientNetV2: Smaller Models and
        Faster Training.” Proceedings of the 38th International Conference on
        Machine Learning (ICML), vol. 139, 2021, pp. 10096-10106.

    """

    class MBConv(MobileNetBlock.InvRes_SE):
        """
        Mobile block used in EfficientNet architecture. It is the same with
        `MobileNetBlock.InvRes_SE`.

        Parameters
        ----------
        `se_reduction` : float, default=4
            Reduction factor for SE block

        Refer to the figures shown in the original paper[1].
        """

    class FusedMBConv(efficient._FusedMBConv):
        """
        Fused Mobile block used in EfficientNet-v2 architecture.

        Parameters
        ----------
        `se_reduction` : float, default=4
            Reduction factor for SE block

        Refer to the figures shown in the original paper[2].
        """


class ResNeStBlock(resnest._ResNeStBlock):
    """
    ResNeSt block which enhances the original ResNet's Bottleneck module
    by integrating Split-Attention mechanism, used in ResNeSt architecture.

    Parameters
    ----------
    `n_splits` : int, default=2
        Number of the splits
    `n_groups` : int, default=32
        Number of the cardinality applied in Split phase
    `reduction` : int, default=4
        Reduction rate for squeezing the dimension of the first FC-layer
        during Attention phase

    Notes
    -----
    - This block has an internal Bottleneck-like mechanism, meaning that
    it handles downsampling strategy automatically based on its channels
    and stride information, according to the original paper[1].

    Reference
    ---------
    `ResNeSt-(50, 101, 200, 269)` :
        [1] Zhang, Hang, et al. "ResNeSt: Split-Attention Networks." arXiv
        preprint arXiv:2004.08955 (2020).

    """


class ConvNeXtBlock(convnext._ConvNeXtBlock):
    """
    A ConvNeXt Block is a building block of the ConvNeXt architecture,
    designed to improve traditional convolutional layers by incorporating a
    depthwise convolution followed by layer normalization, a pointwise
    convolution, and a GELU activation, along with residual connections for
    enhanced feature extraction and better gradient flow.

    This block aims to bridge the gap between convolutional networks and
    transformer-based architectures in terms of performance and scalability.

    Reference
    ---------
    `ConvNeXt-(T, S, B, L, XL)` :
        [1] Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the
        IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
        2022, pp. 11976-11986.

    """
