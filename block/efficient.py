from typing import Tuple, Callable
from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode

from .se import _SEBlock2D


class _FusedMBConv(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int] | int = 3,
        stride: int = 1,
        expand: float = 1.0,
        se_reduction: int = 4,
        activation: Callable = nl.Activation.Swish,
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

        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )

        assert self.stride in [1, 2]
        self.do_shortcut = stride == 1 and in_channels == out_channels
        self.hid_channels = int(round(in_channels * self.expand))

        self.init_nodes()
        super(_FusedMBConv, self).__init__(
            graph={
                self.rt_: [self.conv_],
                self.conv_: [self.se_block, self.scale_],
                self.se_block: [self.scale_],
                self.scale_: [self.tm_],
            },
            root=self.rt_,
            term=self.tm_,
        )

        if self.do_shortcut:
            self[self.rt_].append(self.tm_)

        if self.expand != 1:
            self[self.rt_].remove(self.conv_)
            self[self.rt_].append(self.conv_exp)

            del self.graph[self.conv_]
            self.graph[self.conv_exp] = [self.se_block, self.scale_]

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")

        self.conv_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(
                    self.in_channels,
                    self.out_channels,
                    self.filter_size,
                    self.stride,
                    padding=self.filter_size // 2,
                    **self.basic_args,
                ),
                (
                    nl.BatchNorm2D(self.out_channels, self.momentum)
                    if self.do_batch_norm
                    else None
                ),
                self.activation(),
            ),
            name="conv_",
        )
        self.conv_exp = LayerNode(
            nl.Sequential(
                nl.Conv2D(
                    self.in_channels,
                    self.hid_channels,
                    self.filter_size,
                    self.stride,
                    padding=self.filter_size // 2,
                    **self.basic_args,
                ),
                (
                    nl.BatchNorm2D(self.hid_channels, self.momentum)
                    if self.do_batch_norm
                    else None
                ),
                self.activation(),
                nl.Conv2D(
                    self.hid_channels,
                    self.out_channels,
                    1,
                    padding="valid",
                    **self.basic_args,
                ),
                (
                    nl.BatchNorm2D(self.out_channels, self.momentum)
                    if self.do_batch_norm
                    else None
                ),
                self.activation(),
            ),
            name="conv_exp",
        )

        self.se_block = LayerNode(
            _SEBlock2D(
                self.out_channels,
                self.se_reduction,
                self.activation,
                self.optimizer,
                **self.basic_args,
            ),
            name="se_block",
        )
        self.scale_ = LayerNode(nl.Identity(), MergeMode.HADAMARD, name="scale_")
        self.tm_ = LayerNode(nl.Identity(), MergeMode.SUM, name="tm_")

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        if self.expand == 1:
            return self.conv_.out_shape(in_shape)
        else:
            return self.conv_exp.out_shape(in_shape)
