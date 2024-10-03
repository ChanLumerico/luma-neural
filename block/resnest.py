from typing import Tuple, override, ClassVar

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike, LayerLike
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


class _SplitAttention(LayerGraph): ...


class _Bottleneck(LayerGraph): ...
