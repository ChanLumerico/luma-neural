from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode

from .se import _SEBlock2D


class _FusedMBConv(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int] | int = 3,
        stride: int = 1,
        expand: int = 1,
        se_reduction: int = 4,
        activation: callable = Activation.Swish,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.expand = expand
        self.activation = activation
        self.se_reduction = se_reduction
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum

        self.basic_args = {
            "initializer": initializer,
            "lambda_": lambda_,
            "random_state": random_state,
        }

        assert self.stride in [1, 2]
        self.do_shortcut = stride == 1 and in_channels == out_channels
        self.hid_channels = int(round(in_channels * self.expand))

        self.init_nodes()
        super(_FusedMBConv, self).__init__(
            graph={
                self.rt_: [self.conv_],
                self.conv_: [self.se_, self.scale_],
                self.se_: [self.scale_],
                self.scale_: [self.tm_],
            },
            root=self.rt_,
            term=self.tm_,
        )

        if self.do_shortcut:
            self[self.rt_].append(self.tm_)

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(Identity(), name="rt_")
        self.conv_ = LayerNode(
            Sequential(
                Conv2D(
                    self.in_channels,
                    self.hid_channels,
                    1,
                    padding="valid",
                    **self.basic_args
                ),
                BatchNorm2D(self.hid_channels, self.momentum),
                self.activation(),
                Conv2D(
                    self.hid_channels,
                    self.out_channels,
                    self.filter_size,
                    self.stride,
                    self.filter_size // 2,
                    **self.basic_args
                ),
                BatchNorm2D(self.out_channels, self.momentum),
            ),
            name="conv_",
        )
        self.se_ = LayerNode(
            _SEBlock2D(self.out_channels, self.se_reduction, **self.basic_args),
            name="se_",
        )
        self.scale_ = LayerNode(Identity(), MergeMode.HADAMARD, name="scales_")
        self.tm_ = LayerNode(Identity(), MergeMode.SUM, name="tm_")

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return self.conv_.out_shape(in_shape)
