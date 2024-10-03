from dataclasses import asdict
from typing import Any, Self, override, ClassVar

from luma.core.super import Estimator, Evaluator
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl
from luma.neural import functional as F

from ..types import ImageClassifier

Bottleneck = nb.ResNetBlock.Bottleneck
Bottleneck_SE = nb.ResNetBlock.Bottleneck_SE
Bottleneck_SK = nb.ResNetBlock.Bottleneck_SK


class _ResNeXt_50(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        cardinality: int = 32,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
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
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 128, 256] * 3,
            [256, 256, 512] * 4,
            [512, 512, 1024] * 6,
            [1024, 1024, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "cardinality": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )
        res_args["groups"] = self.cardinality

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        layer_label = "ResNeXtConv"
        Bottleneck.override_expansion(value=2)
        self.layer_2, in_channels = F.make_res_layers(
            64, 128, Bottleneck, 3, 2, base_args, res_args, 1, layer_label
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck, 4, 3, base_args, res_args, 2, layer_label
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck, 6, 4, base_args, res_args, 2, layer_label
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 1024, Bottleneck, 3, 5, base_args, res_args, 2, layer_label
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            nl.AdaptiveAvgPool2D((1, 1)),
            nl.Flatten(),
            nl.Dense(1024 * Bottleneck.expansion, self.out_features, **base_args),
        )
        Bottleneck.reset_expansion()

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNeXt_50, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNeXt_50, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_ResNeXt_50, self).score_nn(X, y, metric, argmax)


class _ResNeXt_101(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        cardinality: int = 32,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
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
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 128, 256] * 3,
            [256, 256, 512] * 4,
            [512, 512, 1024] * 23,
            [1024, 1024, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "cardinality": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )
        res_args["groups"] = self.cardinality

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        layer_label = "ResNeXtConv"
        Bottleneck.override_expansion(value=2)
        self.layer_2, in_channels = F.make_res_layers(
            64, 128, Bottleneck, 3, 2, base_args, res_args, 1, layer_label
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck, 4, 3, base_args, res_args, 2, layer_label
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck, 23, 4, base_args, res_args, 2, layer_label
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 1024, Bottleneck, 3, 5, base_args, res_args, 2, layer_label
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            nl.AdaptiveAvgPool2D((1, 1)),
            nl.Flatten(),
            nl.Dense(1024 * Bottleneck.expansion, self.out_features, **base_args),
        )
        Bottleneck.reset_expansion()

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ResNeXt_101, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNeXt_101, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_ResNeXt_101, self).score_nn(X, y, metric, argmax)


class _SE_ResNeXt_50(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        cardinality: int = 32,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
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
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 128, 256] * 3,
            [256, 256, 512] * 4,
            [512, 512, 1024] * 6,
            [1024, 1024, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "cardinality": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )
        res_args["groups"] = self.cardinality

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        layer_label = "ResNeXtConv"
        Bottleneck_SE.override_expansion(value=2)
        self.layer_2, in_channels = F.make_res_layers(
            64, 128, Bottleneck_SE, 3, 2, base_args, res_args, 1, layer_label
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SE, 4, 3, base_args, res_args, 2, layer_label
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SE, 6, 4, base_args, res_args, 2, layer_label
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 1024, Bottleneck_SE, 3, 5, base_args, res_args, 2, layer_label
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            nl.AdaptiveAvgPool2D((1, 1)),
            nl.Flatten(),
            nl.Dense(1024 * Bottleneck_SE.expansion, self.out_features, **base_args),
        )
        Bottleneck_SE.reset_expansion()

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_SE_ResNeXt_50, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_SE_ResNeXt_50, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_SE_ResNeXt_50, self).score_nn(X, y, metric, argmax)


class _SE_ResNeXt_101(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        cardinality: int = 32,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
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
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 128, 256] * 3,
            [256, 256, 512] * 4,
            [512, 512, 1024] * 23,
            [1024, 1024, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "cardinality": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )
        res_args["groups"] = self.cardinality

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        layer_label = "ResNeXtConv"
        Bottleneck_SE.override_expansion(value=2)
        self.layer_2, in_channels = F.make_res_layers(
            64, 128, Bottleneck_SE, 3, 2, base_args, res_args, 1, layer_label
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SE, 4, 3, base_args, res_args, 2, layer_label
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SE, 23, 4, base_args, res_args, 2, layer_label
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 1024, Bottleneck_SE, 3, 5, base_args, res_args, 2, layer_label
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            nl.AdaptiveAvgPool2D((1, 1)),
            nl.Flatten(),
            nl.Dense(1024 * Bottleneck_SE.expansion, self.out_features, **base_args),
        )
        Bottleneck_SE.reset_expansion()

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_SE_ResNeXt_101, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_SE_ResNeXt_101, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_SE_ResNeXt_101, self).score_nn(X, y, metric, argmax)


class _SK_ResNeXt_50(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        cardinality: int = 32,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
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
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 128, 256] * 3,
            [256, 256, 512] * 4,
            [512, 512, 1024] * 6,
            [1024, 1024, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "cardinality": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )
        res_args["groups"] = self.cardinality

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        layer_label = "ResNeXtConv"
        Bottleneck_SK.override_expansion(value=2)
        self.layer_2, in_channels = F.make_res_layers(
            64, 128, Bottleneck_SK, 3, 2, base_args, res_args, 1, layer_label
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SK, 4, 3, base_args, res_args, 2, layer_label
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SK, 6, 4, base_args, res_args, 2, layer_label
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 1024, Bottleneck_SK, 3, 5, base_args, res_args, 2, layer_label
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            nl.AdaptiveAvgPool2D((1, 1)),
            nl.Flatten(),
            nl.Dense(1024 * Bottleneck_SK.expansion, self.out_features, **base_args),
        )
        Bottleneck_SK.reset_expansion()

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_SK_ResNeXt_50, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_SK_ResNeXt_50, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_SK_ResNeXt_50, self).score_nn(X, y, metric, argmax)


class _SK_ResNeXt_101(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        cardinality: int = 32,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
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
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 128, 256] * 3,
            [256, 256, 512] * 4,
            [512, 512, 1024] * 23,
            [1024, 1024, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.set_param_ranges(
            {
                "cardinality": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "momentum": ("0,1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": ("0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )
        res_args["groups"] = self.cardinality

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        layer_label = "ResNeXtConv"
        Bottleneck_SK.override_expansion(value=2)
        self.layer_2, in_channels = F.make_res_layers(
            64, 128, Bottleneck_SK, 3, 2, base_args, res_args, 1, layer_label
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SK, 4, 3, base_args, res_args, 2, layer_label
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SK, 23, 4, base_args, res_args, 2, layer_label
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 1024, Bottleneck_SK, 3, 5, base_args, res_args, 2, layer_label
        )

        self.model.extend(
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.layer_5,
            deep_add=True,
        )
        self.model.extend(
            nl.AdaptiveAvgPool2D((1, 1)),
            nl.Flatten(),
            nl.Dense(1024 * Bottleneck_SK.expansion, self.out_features, **base_args),
        )
        Bottleneck_SK.reset_expansion()

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_SK_ResNeXt_101, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_SK_ResNeXt_101, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_SK_ResNeXt_101, self).score_nn(X, y, metric, argmax)
