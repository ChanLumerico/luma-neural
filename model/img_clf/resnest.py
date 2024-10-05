from typing import Any, Self, override, ClassVar

from luma.core.super import Estimator, Evaluator
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.preprocessing.encoder import LabelSmoothing
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl

from ..types import ImageClassifier


class _ResNeSt_50(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        radix: int = 2,
        cardinality: int = 1,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0001,
        momentum: float = 0.9,
        dropout_rate: float = 0.2,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.radix = radix
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
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
            [64, 64, 256] * 3,
            [128, 128, 512] * 4,
            [256, 256, 1024] * 6,
            [512, 512, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

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
        res_args = dict(
            n_splits=self.radix,
            n_groups=self.cardinality,
            activation=self.activation,
            **base_args,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 32, 3, 1, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 64, 3, 1, "same", **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
        )

        in_ = 64
        res_blocks_arr = [3, 4, 6, 3]
        out_channels_arr = [64, 128, 256, 512]

        for i, (n_blocks, out) in enumerate(zip(res_blocks_arr, out_channels_arr)):
            for j in range(n_blocks):
                stride = 2 if j == 0 else 1
                self.model += (
                    f"ResNeStConv{i + 1}_{j + 1}",
                    nb.ResNeStBlock(in_, out, stride=stride, **res_args),
                )
                in_ = out * nb.ResNeStBlock.expansion

        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(512 * nb.ResNeStBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_ResNeSt_50, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNeSt_50, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNeSt_50, self).score_nn(X, y, metric, argmax)


class _ResNeSt_101(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        radix: int = 2,
        cardinality: int = 1,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0001,
        momentum: float = 0.9,
        dropout_rate: float = 0.2,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.radix = radix
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
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
            [64, 64, 256] * 3,
            [128, 128, 512] * 4,
            [256, 256, 1024] * 23,
            [512, 512, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

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
        res_args = dict(
            n_splits=self.radix,
            n_groups=self.cardinality,
            activation=self.activation,
            **base_args,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 32, 3, 1, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 64, 3, 1, "same", **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
        )

        in_ = 64
        res_blocks_arr = [3, 4, 23, 3]
        out_channels_arr = [64, 128, 256, 512]

        for i, (n_blocks, out) in enumerate(zip(res_blocks_arr, out_channels_arr)):
            for j in range(n_blocks):
                stride = 2 if j == 0 else 1
                self.model += (
                    f"ResNeStConv{i + 1}_{j + 1}",
                    nb.ResNeStBlock(in_, out, stride=stride, **res_args),
                )
                in_ = out * nb.ResNeStBlock.expansion

        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(512 * nb.ResNeStBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_ResNeSt_101, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNeSt_101, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNeSt_101, self).score_nn(X, y, metric, argmax)


class _ResNeSt_200(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        radix: int = 2,
        cardinality: int = 1,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0001,
        momentum: float = 0.9,
        dropout_rate: float = 0.2,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.radix = radix
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
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
            [64, 64, 256] * 3,
            [128, 128, 512] * 24,
            [256, 256, 1024] * 36,
            [512, 512, 2048] * 3,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

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
        res_args = dict(
            n_splits=self.radix,
            n_groups=self.cardinality,
            activation=self.activation,
            **base_args,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 32, 3, 1, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 64, 3, 1, "same", **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
        )

        in_ = 64
        res_blocks_arr = [3, 24, 36, 3]
        out_channels_arr = [64, 128, 256, 512]

        for i, (n_blocks, out) in enumerate(zip(res_blocks_arr, out_channels_arr)):
            for j in range(n_blocks):
                stride = 2 if j == 0 else 1
                self.model += (
                    f"ResNeStConv{i + 1}_{j + 1}",
                    nb.ResNeStBlock(in_, out, stride=stride, **res_args),
                )
                in_ = out * nb.ResNeStBlock.expansion

        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(512 * nb.ResNeStBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_ResNeSt_200, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNeSt_200, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNeSt_200, self).score_nn(X, y, metric, argmax)


class _ResNeSt_269(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        radix: int = 2,
        cardinality: int = 1,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0001,
        momentum: float = 0.9,
        dropout_rate: float = 0.2,
        smoothing: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.radix = radix
        self.cardinality = cardinality
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.smoothing = smoothing
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
            [64, 64, 256] * 3,
            [128, 128, 512] * 30,
            [256, 256, 1024] * 48,
            [512, 512, 2048] * 8,
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

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
        res_args = dict(
            n_splits=self.radix,
            n_groups=self.cardinality,
            activation=self.activation,
            **base_args,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 32, 3, 1, "same", **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
            nl.Conv2D(32, 64, 3, 1, "same", **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
        )

        in_ = 64
        res_blocks_arr = [3, 30, 48, 8]
        out_channels_arr = [64, 128, 256, 512]

        for i, (n_blocks, out) in enumerate(zip(res_blocks_arr, out_channels_arr)):
            for j in range(n_blocks):
                stride = 2 if j == 0 else 1
                self.model += (
                    f"ResNeStConv{i + 1}_{j + 1}",
                    nb.ResNeStBlock(in_, out, stride=stride, **res_args),
                )
                in_ = out * nb.ResNeStBlock.expansion

        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(512 * nb.ResNeStBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_ResNeSt_269, self).fit_nn(X, y_ls)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_ResNeSt_269, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_ResNeSt_269, self).score_nn(X, y, metric, argmax)
