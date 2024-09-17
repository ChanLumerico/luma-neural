from typing import Self, override, ClassVar, List

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural.block import MobileNetBlock
from luma.neural.layer import *

MBConv = MobileNetBlock.InvRes_SE
MBConv.__name__ = "MBConv"


class _EfficientNet_B0(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B1(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B2(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B3(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B4(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B5(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B6(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B7(Estimator, Supervised, NeuralModel): ...
