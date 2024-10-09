from typing import Any, Self, override, ClassVar

from luma.core.super import Estimator, Evaluator
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl

from ..types import ImageClassifier


class _ConvNeXt_T(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.GELU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
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

        self.set_param_ranges(
            {
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

        depths = [3, 3, 9, 3]
        channels = [96, 192, 384, 768]
        self.model.extend(
            nl.Conv2D(3, channels[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(channels)):
            for j in range(depths[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock(channels[i], 7, self.activation, **base_args),
                )

            if i < len(channels) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(channels[i], channels[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(channels[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ConvNeXt_T, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ConvNeXt_T, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_ConvNeXt_T, self).score_nn(X, y, metric, argmax)


class _ConvNeXt_S(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.GELU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
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

        self.set_param_ranges(
            {
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

        depths = [3, 3, 27, 3]
        channels = [96, 192, 384, 768]
        self.model.extend(
            nl.Conv2D(3, channels[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(channels)):
            for j in range(depths[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock(channels[i], 7, self.activation, **base_args),
                )

            if i < len(channels) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(channels[i], channels[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(channels[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ConvNeXt_S, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ConvNeXt_S, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_ConvNeXt_S, self).score_nn(X, y, metric, argmax)


class _ConvNeXt_B(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.GELU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
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

        self.set_param_ranges(
            {
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

        depths = [3, 3, 27, 3]
        channels = [128, 256, 512, 1024]
        self.model.extend(
            nl.Conv2D(3, channels[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(channels)):
            for j in range(depths[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock(channels[i], 7, self.activation, **base_args),
                )

            if i < len(channels) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(channels[i], channels[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(channels[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ConvNeXt_B, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ConvNeXt_B, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_ConvNeXt_B, self).score_nn(X, y, metric, argmax)


class _ConvNeXt_L(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.GELU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
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

        self.set_param_ranges(
            {
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

        depths = [3, 3, 27, 3]
        channels = [192, 384, 768, 1536]
        self.model.extend(
            nl.Conv2D(3, channels[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(channels)):
            for j in range(depths[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock(channels[i], 7, self.activation, **base_args),
                )

            if i < len(channels) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(channels[i], channels[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(channels[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ConvNeXt_L, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ConvNeXt_L, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_ConvNeXt_L, self).score_nn(X, y, metric, argmax)


class _ConvNeXt_XL(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.GELU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
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

        self.set_param_ranges(
            {
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

        depths = [3, 3, 27, 3]
        channels = [256, 512, 1024, 2048]
        self.model.extend(
            nl.Conv2D(3, channels[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(channels)):
            for j in range(depths[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock(channels[i], 7, self.activation, **base_args),
                )

            if i < len(channels) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(channels[i], channels[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(channels[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_ConvNeXt_XL, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ConvNeXt_XL, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator = Accuracy, argmax: bool = True
    ) -> float:
        return super(_ConvNeXt_XL, self).score_nn(X, y, metric, argmax)
