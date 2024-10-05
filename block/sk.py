from typing import override, Dict, List

from luma.core.super import Optimizer
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _SKBlock1D(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_sizes: list[int] = [3, 5],
        reduction: int = 16,
        groups: int = 1,
        activation: callable = nl.Activation.ReLU,
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
        self.groups = groups
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

        super(_SKBlock1D, self).__init__(
            graph=self._build_graph(), root=self.rt_, term=self.sum_
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.br_arr: list[LayerNode] = []
        self.scale_arr: list[LayerNode] = []
        self.slice_arr: list[LayerNode] = []

        self.rt_ = LayerNode(nl.Identity(), name="rt_")

        for i, f_size in enumerate(self.filter_sizes):
            branch = LayerNode(
                nl.Sequential(
                    nl.Conv1D(
                        self.in_channels,
                        self.out_channels,
                        filter_size=f_size,
                        groups=self.groups,
                        padding="same",
                        **self.basic_args,
                    ),
                    nl.BatchNorm1D(self.out_channels, self.momentum),
                    self.activation(),
                ),
                name=f"branch_{i}",
            )
            self.br_arr.append(branch)

            scale = LayerNode(nl.Identity(), MergeMode.HADAMARD, name=f"scale_{i}")
            self.scale_arr.append(scale)

            slice_ = LayerNode(nl.Slice(f":, {i}, :, :"), name=f"slice_{i}")
            self.slice_arr.append(slice_)

        self.fc_ = LayerNode(
            nl.Sequential(
                nl.GlobalAvgPool1D(),
                nl.Conv1D(
                    self.out_channels,
                    self.out_channels // self.reduction,
                    filter_size=1,
                    padding="valid",
                    **self.basic_args,
                ),
                self.activation(),
                nl.Conv1D(
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
            nl.Sequential(
                nl.Reshape(-1, self.n_branch, self.out_channels),
                nl.Activation.Softmax(dim=1),
                nl.Reshape(-1, self.n_branch, self.out_channels, 1),
            ),
            name="softmax_",
        )
        self.sum_ = LayerNode(nl.Identity(), MergeMode.SUM, name="sum_")

    def _build_graph(self) -> Dict[LayerNode, List[LayerNode]]:
        graph = {}
        graph[self.rt_] = self.br_arr

        for br, sc in zip(self.br_arr, self.scale_arr):
            graph[br] = [self.fc_, sc]

        graph[self.fc_] = [self.softmax_]
        graph[self.softmax_] = self.slice_arr

        for sl, sc in zip(self.slice_arr, self.scale_arr):
            graph[sl] = [sc]
            graph[sc] = [self.sum_]

        return graph

    @Tensor.force_dim(3)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        return super().forward(X, is_train)

    @Tensor.force_dim(3)
    def backward(self, d_out: Tensor) -> Tensor:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        batch_size, _, width = in_shape
        return batch_size, self.out_channels, width


class _SKBlock2D(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_sizes: list[int] = [3, 5],
        reduction: int = 16,
        groups: int = 1,
        activation: callable = nl.Activation.ReLU,
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
        self.groups = groups
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

        super(_SKBlock2D, self).__init__(
            graph=self._build_graph(), root=self.rt_, term=self.sum_
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.br_arr: list[LayerNode] = []
        self.scale_arr: list[LayerNode] = []
        self.slice_arr: list[LayerNode] = []

        self.rt_ = LayerNode(nl.Identity(), name="rt_")

        for i, f_size in enumerate(self.filter_sizes):
            branch = LayerNode(
                nl.Sequential(
                    nl.Conv2D(
                        self.in_channels,
                        self.out_channels,
                        filter_size=f_size,
                        groups=self.groups,
                        padding="same",
                        **self.basic_args,
                    ),
                    nl.BatchNorm2D(self.out_channels, self.momentum),
                    self.activation(),
                ),
                name=f"branch_{i}",
            )
            self.br_arr.append(branch)

            scale = LayerNode(nl.Identity(), MergeMode.HADAMARD, name=f"scale_{i}")
            self.scale_arr.append(scale)

            slice_ = LayerNode(nl.Slice(f":, {i}, :, :, :"), name=f"slice_{i}")
            self.slice_arr.append(slice_)

        self.fc_ = LayerNode(
            nl.Sequential(
                nl.GlobalAvgPool2D(),
                nl.Conv2D(
                    self.out_channels,
                    self.out_channels // self.reduction,
                    filter_size=1,
                    padding="valid",
                    **self.basic_args,
                ),
                self.activation(),
                nl.Conv2D(
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
            nl.Sequential(
                nl.Reshape(-1, self.n_branch, self.out_channels),
                nl.Activation.Softmax(dim=1),
                nl.Reshape(-1, self.n_branch, self.out_channels, 1, 1),
            ),
            name="softmax_",
        )
        self.sum_ = LayerNode(nl.Identity(), MergeMode.SUM, name="sum_")

    def _build_graph(self) -> Dict[LayerNode, List[LayerNode]]:
        graph = {}
        graph[self.rt_] = self.br_arr

        for br, sc in zip(self.br_arr, self.scale_arr):
            graph[br] = [self.fc_, sc]

        graph[self.fc_] = [self.softmax_]
        graph[self.softmax_] = self.slice_arr

        for sl, sc in zip(self.slice_arr, self.scale_arr):
            graph[sl] = [sc]
            graph[sc] = [self.sum_]

        return graph

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        return super().forward(X, is_train)

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        batch_size, _, height, width = in_shape
        return batch_size, self.out_channels, height, width


class _SKBlock3D(LayerGraph):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_sizes: list[int] = [3, 5],
        reduction: int = 16,
        groups: int = 1,
        activation: callable = nl.Activation.ReLU,
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
        self.groups = groups
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

        super(_SKBlock3D, self).__init__(
            graph=self._build_graph(), root=self.rt_, term=self.sum_
        )

        self.build()
        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.br_arr: list[LayerNode] = []
        self.scale_arr: list[LayerNode] = []
        self.slice_arr: list[LayerNode] = []

        self.rt_ = LayerNode(nl.Identity(), name="rt_")

        for i, f_size in enumerate(self.filter_sizes):
            branch = LayerNode(
                nl.Sequential(
                    nl.Conv3D(
                        self.in_channels,
                        self.out_channels,
                        groups=self.groups,
                        filter_size=f_size,
                        padding="same",
                        **self.basic_args,
                    ),
                    nl.BatchNorm3D(self.out_channels, self.momentum),
                    self.activation(),
                ),
                name=f"branch_{i}",
            )
            self.br_arr.append(branch)

            scale = LayerNode(nl.Identity(), MergeMode.HADAMARD, name=f"scale_{i}")
            self.scale_arr.append(scale)

            slice_ = LayerNode(nl.Slice(f":, {i}, :, :, :, :"), name=f"slice_{i}")
            self.slice_arr.append(slice_)

        self.fc_ = LayerNode(
            nl.Sequential(
                nl.GlobalAvgPool3D(),
                nl.Conv3D(
                    self.out_channels,
                    self.out_channels // self.reduction,
                    filter_size=1,
                    padding="valid",
                    **self.basic_args,
                ),
                self.activation(),
                nl.Conv3D(
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
            nl.Sequential(
                nl.Reshape(-1, self.n_branch, self.out_channels),
                nl.Activation.Softmax(dim=1),
                nl.Reshape(-1, self.n_branch, self.out_channels, 1, 1, 1),
            ),
            name="softmax_",
        )
        self.sum_ = LayerNode(nl.Identity(), MergeMode.SUM, name="sum_")

    def _build_graph(self) -> Dict[LayerNode, List[LayerNode]]:
        graph = {}
        graph[self.rt_] = self.br_arr

        for br, sc in zip(self.br_arr, self.scale_arr):
            graph[br] = [self.fc_, sc]

        graph[self.fc_] = [self.softmax_]
        graph[self.softmax_] = self.slice_arr

        for sl, sc in zip(self.slice_arr, self.scale_arr):
            graph[sl] = [sc]
            graph[sc] = [self.sum_]

        return graph

    @Tensor.force_dim(5)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        return super().forward(X, is_train)

    @Tensor.force_dim(5)
    def backward(self, d_out: Tensor) -> Tensor:
        return super().backward(d_out)

    @override
    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        batch_size, _, depth, height, width = in_shape
        return batch_size, self.out_channels, depth, height, width
