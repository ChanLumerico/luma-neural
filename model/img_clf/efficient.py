from typing import Self, override, ClassVar

from luma.core.super import Estimator, Evaluator, Supervised
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural.block import MobileNetBlock
from luma.neural.layer import *
from luma.neural import functional as F

MBConv = MobileNetBlock.InvRes_SE
MBConv.__name__ = "MBConv"

b0_config = [
    [16, 1, 1, 1, 3],
    [24, 2, 6, 2, 3],
    [40, 2, 6, 2, 5],
    [80, 3, 6, 2, 3],
    [112, 3, 6, 1, 5],
    [192, 4, 6, 2, 5],
    [320, 1, 6, 1, 3],
]

# TODO: this is temporary
multipliers = [
    [1.00, 1.00, 1.00],
    [1.20, 1.10, 1.15],
    [1.44, 1.21, 1.32],
    [1.72, 1.33, 1.52],
    ...,
]


class _EfficientNet_B0(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.Swish,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        growth_rate: float = 32,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.growth_rate = growth_rate
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model: Sequential = Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
                "growth_rate": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=0,
        )

        self.model.extend(
            Conv2D(3, 32, 3, 2, **base_args),
            BatchNorm2D(32, self.momentum),
            self.activation(),
        )

        in_ = 32
        for i, (out, n_layers, exp, s, f) in enumerate(mbconv_config):
            for j in range(n_layers):
                s_ = s if j == 0 else 1
                self.model += (
                    f"MBConv{i + 1}_{j + 1}",
                    MBConv(in_, out, f, s_, exp, 4, self.activation, **base_args),
                )
                in_ = out

        dense_in_features = int(round(1280 * multipliers[0][0]))
        self.model.extend(
            Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            GlobalAvgPool2D(),
            Flatten(),
            Dense(dense_in_features, self.out_features, **base_args),
        )

    input_size: ClassVar[tuple] = F.get_efficient_net_input_size(multipliers, n=0)
    input_shape: ClassVar[tuple] = (-1, 3, *input_size)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_EfficientNet_B0, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_EfficientNet_B0, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_EfficientNet_B0, self).score_nn(X, y, metric, argmax)


class _EfficientNet_B1(Estimator, Supervised, NeuralModel):
    def __init__(
        self,
        activation: callable = Activation.Swish,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        growth_rate: float = 32,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.growth_rate = growth_rate
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model: Sequential = Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
                "growth_rate": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = {
            "initializer": self.initializer,
            "lambda_": self.lambda_,
            "random_state": self.random_state,
        }
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=1,
        )

        self.model.extend(
            Conv2D(3, 32, 3, 2, **base_args),
            BatchNorm2D(32, self.momentum),
            self.activation(),
        )

        in_ = 32
        for i, (out, n_layers, exp, s, f) in enumerate(mbconv_config):
            for j in range(n_layers):
                s_ = s if j == 0 else 1
                self.model += (
                    f"MBConv{i + 1}_{j + 1}",
                    MBConv(in_, out, f, s_, exp, 4, self.activation, **base_args),
                )
                in_ = out

        dense_in_features = int(round(1280 * multipliers[1][0]))
        self.model.extend(
            Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            GlobalAvgPool2D(),
            Flatten(),
            Dense(dense_in_features, self.out_features, **base_args),
        )

    input_size: ClassVar[tuple] = F.get_efficient_net_input_size(multipliers, n=1)
    input_shape: ClassVar[tuple] = (-1, 3, *input_size)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_EfficientNet_B1, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_EfficientNet_B1, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_EfficientNet_B1, self).score_nn(X, y, metric, argmax)


class _EfficientNet_B2(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B3(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B4(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B5(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B6(Estimator, Supervised, NeuralModel): ...


class _EfficientNet_B7(Estimator, Supervised, NeuralModel): ...
