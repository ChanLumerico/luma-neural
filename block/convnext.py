from typing import Dict, List, override, ClassVar

from luma.core.super import Optimizer
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode

from .resnet import _ExpansionMixin


class ConvNeXtBlock(LayerGraph): ...
