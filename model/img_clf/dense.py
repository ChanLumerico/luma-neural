from typing import Self, override, ClassVar, List

from luma.core.super import Estimator, Evaluator
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl
from luma.neural import functional as F

from ..types import ImageClassifier

Composite = nb.DenseNetBlock.Composite
DenseUnit = nb.DenseNetBlock.DenseUnit
Transition = nb.DenseNetBlock.Transition


class _DenseNet_121(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
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
        self.model: nl.Sequential = nl.Sequential()

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
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        dense_args = dict(**base_args, momentum=self.momentum)

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", 1),
        )

        in_channels = 64
        n_layers_arr = [6, 12, 24, 16]
        for i, n_layers in enumerate(n_layers_arr):
            self.model += (
                f"DenseBlock_{i + 1}",
                DenseUnit(in_channels, n_layers, self.growth_rate, **dense_args),
            )
            in_channels = in_channels + n_layers * self.growth_rate

            if i != len(n_layers_arr) - 1:
                self.model += (
                    f"TransLayer_{i + 1}",
                    Transition(in_channels, in_channels // 2, **dense_args),
                )
                in_channels //= 2

        self.model += nl.BatchNorm2D(in_channels, self.momentum)
        self.model += self.activation()

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(in_channels, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_DenseNet_121, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_DenseNet_121, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_DenseNet_121, self).score_nn(X, y, metric, argmax)


class _DenseNet_169(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
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
        self.model: nl.Sequential = nl.Sequential()

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
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        dense_args = dict(**base_args, momentum=self.momentum)

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", 1),
        )

        in_channels = 64
        n_layers_arr = [6, 12, 32, 32]
        for i, n_layers in enumerate(n_layers_arr):
            self.model += (
                f"DenseBlock_{i + 1}",
                DenseUnit(in_channels, n_layers, self.growth_rate, **dense_args),
            )
            in_channels = in_channels + n_layers * self.growth_rate

            if i != len(n_layers_arr) - 1:
                self.model += (
                    f"TransLayer_{i + 1}",
                    Transition(in_channels, in_channels // 2, **dense_args),
                )
                in_channels //= 2

        self.model += nl.BatchNorm2D(in_channels, self.momentum)
        self.model += self.activation()

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(in_channels, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_DenseNet_169, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_DenseNet_169, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_DenseNet_169, self).score_nn(X, y, metric, argmax)


class _DenseNet_201(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
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
        self.model: nl.Sequential = nl.Sequential()

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
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        dense_args = dict(**base_args, momentum=self.momentum)

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", 1),
        )

        in_channels = 64
        n_layers_arr = [6, 12, 48, 32]
        for i, n_layers in enumerate(n_layers_arr):
            self.model += (
                f"DenseBlock_{i + 1}",
                DenseUnit(in_channels, n_layers, self.growth_rate, **dense_args),
            )
            in_channels = in_channels + n_layers * self.growth_rate

            if i != len(n_layers_arr) - 1:
                self.model += (
                    f"TransLayer_{i + 1}",
                    Transition(in_channels, in_channels // 2, **dense_args),
                )
                in_channels //= 2

        self.model += nl.BatchNorm2D(in_channels, self.momentum)
        self.model += self.activation()

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(in_channels, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_DenseNet_201, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_DenseNet_201, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_DenseNet_201, self).score_nn(X, y, metric, argmax)


class _DenseNet_264(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
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
        self.model: nl.Sequential = nl.Sequential()

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
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        dense_args = dict(**base_args, momentum=self.momentum)

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", 1),
        )

        in_channels = 64
        n_layers_arr = [6, 12, 64, 48]
        for i, n_layers in enumerate(n_layers_arr):
            self.model += (
                f"DenseBlock_{i + 1}",
                DenseUnit(in_channels, n_layers, self.growth_rate, **dense_args),
            )
            in_channels = in_channels + n_layers * self.growth_rate

            if i != len(n_layers_arr) - 1:
                self.model += (
                    f"TransLayer_{i + 1}",
                    Transition(in_channels, in_channels // 2, **dense_args),
                )
                in_channels //= 2

        self.model += nl.BatchNorm2D(in_channels, self.momentum)
        self.model += self.activation()

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(in_channels, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_DenseNet_264, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_DenseNet_264, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_DenseNet_264, self).score_nn(X, y, metric, argmax)


class _SE_DenseNet_121(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
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
        self.model: nl.Sequential = nl.Sequential()

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
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        dense_args = dict(**base_args, momentum=self.momentum)

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", 1),
        )

        in_channels = 64
        n_layers_arr = [6, 12, 24, 16]
        for i, n_layers in enumerate(n_layers_arr):
            dense_se_args = {
                "in_channels": in_channels,
                "n_layers": n_layers,
                "growth_rate": self.growth_rate,
                **dense_args,
            }
            se_args = {"in_channels": in_channels + n_layers * self.growth_rate}

            self.model += (
                f"DenseBlock_SE{i + 1}",
                F.attach_se_block(
                    DenseUnit,
                    nb.SEBlock2D,
                    dense_se_args,
                    se_args,
                ),
            )
            in_channels = in_channels + n_layers * self.growth_rate

            if i != len(n_layers_arr) - 1:
                self.model += (
                    f"TransLayer_{i + 1}",
                    Transition(in_channels, in_channels // 2, **dense_args),
                )
                in_channels //= 2

        self.model += nl.BatchNorm2D(in_channels, self.momentum)
        self.model += self.activation()

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(in_channels, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_SE_DenseNet_121, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_SE_DenseNet_121, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_SE_DenseNet_121, self).score_nn(X, y, metric, argmax)


class _SE_DenseNet_169(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
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
        self.model: nl.Sequential = nl.Sequential()

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
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        dense_args = dict(**base_args, momentum=self.momentum)

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", 1),
        )

        in_channels = 64
        n_layers_arr = [6, 12, 32, 32]
        for i, n_layers in enumerate(n_layers_arr):
            dense_se_args = {
                "in_channels": in_channels,
                "n_layers": n_layers,
                "growth_rate": self.growth_rate,
                **dense_args,
            }
            se_args = {"in_channels": in_channels + n_layers * self.growth_rate}

            self.model += (
                f"DenseBlock_SE{i + 1}",
                F.attach_se_block(
                    DenseUnit,
                    nb.SEBlock2D,
                    dense_se_args,
                    se_args,
                ),
            )
            in_channels = in_channels + n_layers * self.growth_rate

            if i != len(n_layers_arr) - 1:
                self.model += (
                    f"TransLayer_{i + 1}",
                    Transition(in_channels, in_channels // 2, **dense_args),
                )
                in_channels //= 2

        self.model += nl.BatchNorm2D(in_channels, self.momentum)
        self.model += self.activation()

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(in_channels, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_SE_DenseNet_169, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_SE_DenseNet_169, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_SE_DenseNet_169, self).score_nn(X, y, metric, argmax)
