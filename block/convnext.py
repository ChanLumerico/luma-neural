from typing import Dict, List, override, ClassVar

from luma.core.super import Optimizer
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, SequentialNode, LayerGraph, MergeMode

from .resnet import _ExpansionMixin


class _ConvNeXtBlock(LayerGraph, _ExpansionMixin):
    def __init__(
        self,
        in_channels: int,
        filter_size: int = 7,
        activation: callable = nl.Activation.GELU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.momentum = momentum
        self.random_state = random_state

        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

        super(_ConvNeXtBlock, self).__init__(...)

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_node(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")

        exp_channels = self.in_channels * type(self).expansion
        self.conv_ = SequentialNode(
            nl.DepthConv2D(self.in_channels, self.filter_size, **self.basic_args),
            nl.LayerNorm(),
            nl.Conv2D(self.in_channels, exp_channels, 1, **self.basic_args),
            self.activation(),
            nl.Conv2D(exp_channels, self.in_channels, 1, **self.basic_args),
            name="conv_",
        )
