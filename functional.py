from typing import Any, Optional, Tuple, Type, override

from luma.interface.typing import LayerLike, Matrix
from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


def make_res_layers(
    in_channels: int,
    out_channels: int,
    block: Type[LayerLike],
    n_blocks: int,
    layer_num: int,
    conv_base_args: dict,
    res_base_args: dict,
    stride: int = 1,
    layer_label: str = "ResNetConv",
) -> tuple[Sequential, int]:
    if not hasattr(block, "expansion"):
        raise RuntimeError(f"'{block.__name__}' has no expansion factor!")

    downsampling: Optional[Sequential] = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsampling = Sequential(
            Conv2D(
                in_channels,
                out_channels * block.expansion,
                1,
                stride,
                **conv_base_args,
            ),
            BatchNorm2D(out_channels * block.expansion),
        )

    first_block = block(
        in_channels, out_channels, stride, downsampling, **res_base_args
    )
    layers: list = [(f"{layer_label}{layer_num}_1", first_block)]

    in_channels = out_channels * block.expansion
    for i in range(1, n_blocks):
        new_block = (
            f"{layer_label}{layer_num}_{i + 1}",
            block(in_channels, out_channels, **res_base_args),
        )
        layers.append(new_block)

    return Sequential(*layers), in_channels


def attach_se_block(
    layer: type[LayerLike],
    se_block: type[LayerLike],
    layer_args: dict = {},
    se_args: dict = {},
    pre_build: bool = True,
    suffix: str = "SE",
) -> LayerGraph:
    layer_inst = layer(**layer_args)
    se_inst = se_block(**se_args)

    root_node = LayerNode(layer_inst, name="root")
    se_node = LayerNode(se_inst, name="se")
    scale_node = LayerNode(Identity(), MergeMode.HADAMARD, name="scale")

    class _Tmp_LayerGraph(LayerGraph):
        @override
        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return layer_inst.out_shape(in_shape)

    _Tmp_LayerGraph.__name__ = f"{layer.__name__}_{suffix}"

    graph = _Tmp_LayerGraph(
        graph={
            root_node: [se_node, scale_node],
            se_node: [scale_node],
        },
        root=root_node,
        term=scale_node,
    )
    if pre_build:
        graph.build()

    return graph


def get_efficient_net_mbconv_config(
    base_config: list | Matrix, multipliers: list | Matrix, n: int
) -> list:
    if isinstance(base_config, list):
        base_config = Matrix(base_config).astype(float)

    if isinstance(multipliers, list):
        multipliers = Matrix(multipliers)

    new_config = base_config.copy()
    new_config[:, :2] *= multipliers[n]
    new_config = new_config.round().astype(int)

    return new_config.tolist()
