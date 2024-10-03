from typing import Tuple, override, ClassVar

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike, LayerLike
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode

from .se import _SEBlock2D
from .sk import _SKBlock2D


class _ExpansionMixin:
    expansion: ClassVar[int]

    @classmethod
    def check_expansion(cls) -> None:
        if not hasattr(cls, "expansion"):
            raise AttributeError(f"'{cls.__name__}' has no expansion factor!")

    @classmethod
    def override_expansion(cls, value: int) -> None:
        cls.check_expansion()
        cls._original_expansion = cls.expansion
        cls.expansion = value

    @classmethod
    def reset_expansion(cls) -> None:
        cls.check_expansion()
        cls.expansion = cls._original_expansion


class _Basic(LayerGraph, _ExpansionMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsampling = downsampling
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

        super(_Basic, self).__init__(
            graph={
                self.rt_: [self.down_, self.conv_],
                self.conv_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 1

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.conv_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(
                    self.in_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels * type(self).expansion,
                    3,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(
                    self.out_channels * type(self).expansion,
                    self.momentum,
                ),
            ),
            name="conv_",
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else nl.Identity(),
            name="down_",
        )
        self.sum_ = LayerNode(
            self.activation(),
            MergeMode.SUM,
            name="sum_",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)


class _Bottleneck(LayerGraph, _ExpansionMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        groups: int = 1,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsampling = downsampling
        self.groups = groups
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

        super(_Bottleneck, self).__init__(
            graph={
                self.rt_: [self.down_, self.conv_],
                self.conv_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.conv_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(self.in_channels, self.out_channels, 1, **self.basic_args),
                nl.BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    groups=self.groups,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels * type(self).expansion,
                    1,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(
                    self.out_channels * type(self).expansion,
                    self.momentum,
                ),
            ),
            name="conv_",
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else nl.Identity(),
            name="down_",
        )
        self.sum_ = LayerNode(
            self.activation(),
            MergeMode.SUM,
            name="sum_",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)


class _PreActBottleneck(LayerGraph, _ExpansionMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsampling = downsampling
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

        super(_PreActBottleneck, self).__init__(
            graph={
                self.rt_: [self.down_, self.conv_],
                self.conv_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.conv_ = LayerNode(
            nl.Sequential(
                nl.BatchNorm2D(self.in_channels, self.momentum),
                self.activation(),
                nl.Conv2D(
                    self.in_channels,
                    self.out_channels,
                    1,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(self.out_channels),
                self.activation(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels * type(self).expansion,
                    1,
                    **self.basic_args,
                ),
            )
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else nl.Identity(),
            name="down_",
        )
        self.sum_ = LayerNode(
            nl.Identity(),
            MergeMode.SUM,
            name="sum_",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)


class _Bottleneck_SE(LayerGraph, _ExpansionMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        se_reduction: int = 4,
        groups: int = 1,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.se_reduction = se_reduction
        self.groups = groups
        self.downsampling = downsampling
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

        super(_Bottleneck_SE, self).__init__(
            graph={
                self.rt_: [self.conv_, self.down_],
                self.conv_: [self.scale_, self.se_],
                self.se_: [self.scale_],
                self.scale_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.conv_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(self.in_channels, self.out_channels, 1, **self.basic_args),
                nl.BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels,
                    3,
                    self.stride,
                    groups=self.groups,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels * type(self).expansion,
                    1,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(
                    self.out_channels * type(self).expansion,
                    self.momentum,
                ),
            ),
            name="conv_",
        )
        self.se_ = LayerNode(
            _SEBlock2D(
                self.out_channels * type(self).expansion,
                self.se_reduction,
                **self.basic_args,
            ),
            name="se_",
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else nl.Identity(),
            name="down_",
        )
        self.scale_ = LayerNode(
            nl.Identity(),
            MergeMode.HADAMARD,
            name="scale_",
        )
        self.sum_ = LayerNode(
            self.activation(),
            MergeMode.SUM,
            name="sum_",
        )

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)


class _Bottleneck_SK(LayerGraph, _ExpansionMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        filter_sizes: int = [3, 5],
        reduction: int = 16,
        groups: int = 1,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsampling = downsampling
        self.filter_sizes = filter_sizes
        self.reduction = reduction
        self.groups = groups
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

        super(_Bottleneck_SK, self).__init__(
            graph={
                self.rt_: [self.down_, self.conv_],
                self.conv_: [self.sum_],
                self.down_: [self.sum_],
            },
            root=self.rt_,
            term=self.sum_,
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.conv_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(self.in_channels, self.out_channels, 1, **self.basic_args),
                nl.BatchNorm2D(self.out_channels, self.momentum),
                self.activation(),
                _SKBlock2D(
                    self.out_channels,
                    self.out_channels,
                    self.filter_sizes,
                    self.reduction,
                    self.groups,
                    self.activation,
                    **self.basic_args,
                ),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels * type(self).expansion,
                    1,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(
                    self.out_channels * type(self).expansion,
                    self.momentum,
                ),
            ),
            name="conv_",
        )
        self.down_ = LayerNode(
            self.downsampling if self.downsampling else nl.Identity(), name="down_"
        )
        self.sum_ = LayerNode(self.activation(), MergeMode.SUM, name="sum_")

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)
