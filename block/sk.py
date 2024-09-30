from typing import override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class SKBlock(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_sizes: list[int] = [3, 5],
        reduction: int = 16,
        max_branch: int = 32,
        activation: callable = Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.reduction = reduction
        self.max_branch = max_branch
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.momentum = momentum

        self.n_branch = len(filter_sizes)
        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

    def init_nodes(self) -> None:
        self.br_arr: list[LayerNode] = []
        self.scale_arr: list[LayerNode] = []
        self.slice_arr: list[LayerNode] = []

        for i, f_size in enumerate(self.filter_sizes):
            padding = (f_size - 1) // 2
            branch = LayerNode(
                Sequential(
                    Conv2D(
                        self.in_channels,
                        self.out_channels,
                        filter_size=f_size,
                        padding=padding,
                        **self.basic_args,
                    ),
                    BatchNorm2D(self.out_channels, self.momentum),
                    self.activation(),
                ),
                name=f"branch_{i}",
            )
            self.br_arr.append(branch)

            scale = LayerNode(Identity(), MergeMode.HADAMARD, name=f"scale_{i}")
            self.scale_arr.append(scale)

            slice_ = LayerNode(Slice(f":, {i}, :, :, :"), name=f"slice_{i}")
            self.slice_arr.append(slice_)

        self.fc_ = LayerNode(
            Sequential(
                GlobalAvgPool2D(),
                Conv2D(
                    self.out_channels,
                    self.out_channels // self.reduction,
                    filter_size=1,
                    padding="valid",
                    **self.basic_args,
                ),
                self.activation(),
                Conv2D(
                    self.out_channels // self.reduction,
                    self.out_channels * self.n_branch,
                    filter_size=1,
                    padding="valid",
                    **self.basic_args,
                ),
            ),
            merge_mode=MergeMode.SUM,
            name="fc_",
        )

        self.softmax_ = LayerNode(
            Sequential(
                Reshape(-1, self.n_branch, self.out_channels),
                Activation.Softmax(dim=1),
                Reshape(-1, self.n_branch, self.out_channels, 1, 1),
            ),
            name="softmax_",
        )
        self.sum_ = LayerNode(Identity(), MergeMode.SUM, name="sum_")
