from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode

from .se import _SEBlock2D


class _InvRes(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int] | int = 3,
        stride: int = 1,
        expand: int = 1,
        activation: callable = nl.Activation.ReLU6,
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
        super(_InvRes, self).__init__(
            graph={
                self.rt_: [self.dw_],
                self.dw_: [self.tm_],
            },
            root=self.rt_,
            term=self.tm_,
        )

        if self.expand != 1:
            self[self.rt_].clear()
            self[self.rt_].append(self.pw_)
            self.graph[self.pw_] = [self.dw_]

        if self.do_shortcut:
            self[self.rt_].append(self.tm_)

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.pw_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(
                    self.in_channels,
                    self.hid_channels,
                    1,
                    padding="valid",
                    **self.basic_args,
                ),
                (
                    nl.BatchNorm2D(self.hid_channels, self.momentum)
                    if self.do_batch_norm
                    else None
                ),
                self.activation(),
            ),
            name="pw_",
        )
        self.dw_ = LayerNode(
            nl.Sequential(
                nl.DepthConv2D(
                    self.hid_channels,
                    self.filter_size,
                    self.stride,
                    padding="valid" if self.stride == 2 else "same",
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
            ),
            name="dw_",
        )
        self.tm_ = LayerNode(nl.Identity(), MergeMode.SUM, name="tm_")

    @Tensor.force_dim(4)
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: TensorLike) -> TensorLike:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        return (
            batch_size,
            self.out_channels,
            height // self.stride,
            width // self.stride,
        )


class _InvRes_SE(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: Tuple[int, int] | int = 3,
        stride: int = 1,
        expand: int = 1,
        se_reduction: int = 4,
        activation: callable = nl.Activation.HardSwish,
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
        super(_InvRes_SE, self).__init__(
            graph={
                self.rt_: [self.dw_pw_lin],
                self.dw_pw_lin: [self.se_block, self.scale_],
                self.se_block: [self.scale_],
                self.scale_: [self.tm_],
            },
            root=self.rt_,
            term=self.tm_,
        )

        if self.expand != 1:
            self[self.rt_].clear()
            self[self.rt_].append(self.pw_)
            self.graph[self.pw_] = [self.dw_pw_lin]

        if self.do_shortcut:
            self[self.rt_].append(self.tm_)

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.pw_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(
                    self.in_channels,
                    self.hid_channels,
                    1,
                    padding="valid",
                    **self.basic_args,
                ),
                (
                    nl.BatchNorm2D(self.hid_channels, self.momentum)
                    if self.do_batch_norm
                    else None
                ),
                self.activation(),
            ),
            name="pw_",
        )
        self.dw_pw_lin = LayerNode(
            nl.Sequential(
                nl.DepthConv2D(
                    self.hid_channels,
                    self.filter_size,
                    self.stride,
                    padding="valid" if self.stride == 2 else "same",
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
            ),
            name="dw_pw_lin",
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

    @override
    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _, height, width = in_shape
        return (
            batch_size,
            self.out_channels,
            height // self.stride,
            width // self.stride,
        )
