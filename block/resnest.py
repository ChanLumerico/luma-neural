from typing import Tuple, override, ClassVar

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike, LayerLike
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _ResNeStBlock(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        n_splits: int,
        n_groups: int,
        stride: int = 1,
        downsampling: LayerLike | None = None,
        reduction: int = 4,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.n_splits = n_splits
        self.n_groups = n_groups
        self.stride = stride
        self.downsampling = downsampling
        self.reduction = reduction
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

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.downsample_ = LayerNode(
            nl.Identity() if self.downsampling is None else self.downsampling,
            name="downsample_",
        )

        self.split_arr: list[LayerNode] = []
        self.slice_arr: list[LayerNode] = []
        self.scale_arr: list[LayerNode] = []

        for r in range(self.n_splits):
            split = LayerNode(
                nl.Sequential(
                    nl.Conv2D(
                        self.in_channels,
                        self.in_channels // self.n_splits,
                        filter_size=1,
                        padding="valid",
                        groups=self.n_groups,
                        **self.basic_args,
                    ),
                    nl.BatchNorm2D(self.in_channels // self.n_splits, self.momentum),
                    self.activation(),
                    nl.Conv2D(
                        self.in_channels // self.n_splits,
                        self.in_channels,
                        filter_size=3,
                        stride=self.stride,
                        padding="same",
                        groups=self.n_groups,
                        **self.basic_args,
                    ),
                    nl.BatchNorm2D(self.in_channels, self.momentum),
                    self.activation(),
                ),
                name=f"split_{r}",
            )
            self.split_arr.append(split)

            scale = LayerNode(nl.Identity(), MergeMode.HADAMARD, name=f"scale_{r}")
            self.scale_arr.append(scale)

            slice_ = LayerNode(nl.Slice(f":, {r}, :, :, :"), name=f"slice_{r}")
            self.slice_arr.append(slice_)

        self.fc_ = LayerNode(
            nl.Sequential(
                nl.GlobalAvgPool2D(),
                nl.Conv2D(
                    self.in_channels,
                    self.in_channels // self.reduction,
                    filter_size=1,
                    padding="valid",
                    groups=self.n_groups,
                    **self.basic_args,
                ),
                self.activation,
                nl.Conv2D(
                    self.in_channels // self.reduction,
                    self.in_channels * self.n_splits,
                    filter_size=1,
                    padding="valid",
                    groups=self.n_groups,
                    **self.basic_args,
                ),
            ),
            merge_mode=MergeMode.SUM,
            name="fc_",
        )

        self.softmax_ = LayerNode(
            nl.Sequential(
                nl.Reshape(-1, self.n_splits, self.in_channels),
                nl.Activation.Softmax(dim=1),
                nl.Reshape(-1, self.n_splits, self.in_channels, 1, 1),
            ),
            name="softmax_",
        )
        self.sum_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(
                    self.in_channels,
                    self.in_channels,
                    filter_size=1,
                    padding="valid",
                    **self.basic_args,
                ),
                nl.BatchNorm2D(self.in_channels, self.momentum),
                self.activation(),
            ),
            merge_mode=MergeMode.SUM,
            name="sum_",
        )
        self.res_sum = LayerNode(nl.Identity(), MergeMode.SUM, name="res_sum")
