from typing import Dict, List, override, ClassVar

from luma.core.super import Optimizer
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode

from .resnet import _ExpansionMixin


class _ResNeStBlock(LayerGraph, _ExpansionMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_splits: int = 2,
        n_groups: int = 1,
        filter_size: int = 3,
        stride: int = 1,
        reduction: int = 4,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        drop_prob: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_splits = n_splits
        self.n_groups = n_groups
        self.filter_size = filter_size
        self.stride = stride
        self.reduction = reduction
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.momentum = momentum
        self.drop_prob = drop_prob
        self.random_state = random_state

        assert stride in [1, 2]
        self.do_downsample = (
            stride != 1 and in_channels != out_channels * type(self).expansion
        )
        self.basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.init_nodes()

        super(_ResNeStBlock, self).__init__(
            graph=self._build_graph(), root=self.rt_, term=self.res_sum
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    expansion: ClassVar[int] = 4

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")

        self.split_arr: list[LayerNode] = []
        self.slice_arr: list[LayerNode] = []
        self.scale_arr: list[LayerNode] = []

        for r in range(self.n_splits):
            split = LayerNode(
                nl.Sequential(
                    nl.Conv2D(
                        self.in_channels,
                        self.out_channels // self.n_splits,
                        filter_size=1,
                        groups=self.n_groups,
                        **self.basic_args,
                    ),
                    nl.BatchNorm2D(self.out_channels // self.n_splits, self.momentum),
                    self.activation(),
                    nl.Conv2D(
                        self.out_channels // self.n_splits,
                        self.out_channels,
                        self.filter_size,
                        stride=self.stride,
                        groups=self.n_groups,
                        **self.basic_args,
                    ),
                    nl.BatchNorm2D(self.out_channels, self.momentum),
                    nl.DropBlock2D(self.drop_prob, 3, self.random_state),
                    self.activation(),
                ),
                name=f"split_{r}",
            )
            self.split_arr.append(split)

            scale = LayerNode(nl.Identity(), MergeMode.HADAMARD, name=f"scale_{r}")
            self.scale_arr.append(scale)

            slice_ = LayerNode(nl.Slice(f":, {r}, :, :, :"), name=f"slice_{r}")
            self.slice_arr.append(slice_)

        inter_channels = max(self.out_channels // self.reduction, 32)
        self.fc_ = LayerNode(
            nl.Sequential(
                nl.GlobalAvgPool2D(),
                nl.Conv2D(
                    self.out_channels,
                    inter_channels,
                    filter_size=1,
                    groups=self.n_groups,
                    **self.basic_args,
                ),
                self.activation(),
                nl.Conv2D(
                    inter_channels,
                    self.out_channels * self.n_splits,
                    filter_size=1,
                    groups=self.n_groups,
                    **self.basic_args,
                ),
            ),
            merge_mode=MergeMode.SUM,
            name="fc_",
        )

        self.softmax_ = LayerNode(
            nl.Sequential(
                nl.Reshape(-1, self.n_splits, self.out_channels),
                (
                    nl.Activation.Softmax(dim=1)
                    if self.n_splits > 1
                    else nl.Activation.Sigmoid()
                ),
                nl.Reshape(-1, self.n_splits, self.out_channels, 1, 1),
            ),
            name="softmax_",
        )
        self.sum_ = LayerNode(
            nl.Sequential(
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels * type(self).expansion,
                    filter_size=1,
                    **self.basic_args,
                ),
                nl.BatchNorm2D(
                    self.out_channels * type(self).expansion,
                    self.momentum,
                ),
                self.activation(),
            ),
            merge_mode=MergeMode.SUM,
            name="sum_",
        )
        self.res_sum = LayerNode(nl.Identity(), MergeMode.SUM, name="res_sum")

        self.downsample_ = LayerNode(
            (
                nl.Sequential(
                    nl.Pool2D(3, self.stride, "avg", "same"),
                    nl.Conv2D(
                        self.in_channels,
                        self.out_channels * type(self).expansion,
                        filter_size=1,
                        stride=1,
                        **self.basic_args,
                    ),
                    nl.BatchNorm2D(self.out_channels * type(self).expansion),
                )
                if self.do_downsample
                else nl.Identity()
            ),
            name="downsample_",
        )

    def _build_graph(self) -> Dict[LayerNode, List[LayerNode]]:
        graph = {}
        graph[self.rt_] = [*self.split_arr, self.downsample_]

        for r in range(self.n_splits):
            graph[self.split_arr[r]] = [self.fc_, self.scale_arr[r]]
            graph[self.slice_arr[r]] = [self.scale_arr[r]]
            graph[self.scale_arr[r]] = [self.sum_]

        graph[self.fc_] = [self.softmax_]
        graph[self.softmax_] = self.slice_arr

        graph[self.sum_] = [self.res_sum]
        graph[self.downsample_] = [self.res_sum]

        return graph

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        return self.downsample_.out_shape(in_shape)
