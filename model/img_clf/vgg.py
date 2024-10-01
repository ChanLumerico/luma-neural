from typing import Self, override, ClassVar
from dataclasses import asdict

from luma.core.super import Estimator, Evaluator
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil
from luma.metric.classification import Accuracy

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl

from ..types import ImageClassifier


__all__ = (
    "_VGGNet_11",
    "_VGGNet_13",
    "_VGGNet_16",
    "_VGGNet_19",
)


class _VGGNet_11(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
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
        self.dropout_rate = dropout_rate
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
            [3, 64, 128, 256, 256, *[512] * 4],
            [512 * 7 * 7, 4096, 4096, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        conv_3x3_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="max",
            random_state=self.random_state,
        )
        conv_3x3_no_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_pooling=False,
            random_state=self.random_state,
        )
        dense_args = nb.DenseBlockArgs(
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )

        self.model += (
            "ConvBlock_1",
            nb.ConvBlock2D(3, 64, **asdict(conv_3x3_pool_args)),
        )
        self.model += (
            "ConvBlock_2",
            nb.ConvBlock2D(64, 128, **asdict(conv_3x3_pool_args)),
        )
        self.model += (
            "ConvBlock_3",
            nb.ConvBlock2D(128, 256, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_4",
            nb.ConvBlock2D(256, 256, **asdict(conv_3x3_pool_args)),
        )
        self.model += (
            "ConvBlock_5",
            nb.ConvBlock2D(256, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_6",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )
        self.model += (
            "ConvBlock_7",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_8",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )

        self.model += nl.Flatten()
        self.model += (
            "DenseBlock_1",
            nb.DenseBlock(512 * 7 * 7, 4096, **asdict(dense_args)),
        )
        self.model += (
            "DenseBlock_2",
            nb.DenseBlock(4096, 4096, **asdict(dense_args)),
        )
        self.model += nl.Dense(
            4096,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_VGGNet_11, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_VGGNet_11, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_VGGNet_11, self).score_nn(X, y, metric, argmax)


class _VGGNet_13(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
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
        self.dropout_rate = dropout_rate
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
            [3, 64, 64, 128, 128, 256, 256, *[512] * 4],
            [512 * 7 * 7, 4096, 4096, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        conv_3x3_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="max",
            random_state=self.random_state,
        )
        conv_3x3_no_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_pooling=False,
            random_state=self.random_state,
        )
        dense_args = nb.DenseBlockArgs(
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )

        self.model += (
            "ConvBlock_1",
            nb.ConvBlock2D(3, 64, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_2",
            nb.ConvBlock2D(64, 64, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_3",
            nb.ConvBlock2D(64, 128, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_4",
            nb.ConvBlock2D(128, 128, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_5",
            nb.ConvBlock2D(128, 256, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_6",
            nb.ConvBlock2D(256, 256, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_7",
            nb.ConvBlock2D(256, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_8",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_9",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_10",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )

        self.model += nl.Flatten()
        self.model += (
            "DenseBlock_1",
            nb.DenseBlock(512 * 7 * 7, 4096, **asdict(dense_args)),
        )
        self.model += (
            "DenseBlock_2",
            nb.DenseBlock(4096, 4096, **asdict(dense_args)),
        )
        self.model += nl.Dense(
            4096,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_VGGNet_13, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_VGGNet_13, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_VGGNet_13, self).score_nn(X, y, metric, argmax)


class _VGGNet_16(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
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
        self.dropout_rate = dropout_rate
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
            [3, 64, 64, 128, 128, *[256] * 3, *[512] * 6],
            [512 * 7 * 7, 4096, 4096, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        conv_3x3_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="max",
            random_state=self.random_state,
        )
        conv_3x3_no_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_pooling=False,
            random_state=self.random_state,
        )
        dense_args = nb.DenseBlockArgs(
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )

        self.model += (
            "ConvBlock_1",
            nb.ConvBlock2D(3, 64, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_2",
            nb.ConvBlock2D(64, 64, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_3",
            nb.ConvBlock2D(64, 128, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_4",
            nb.ConvBlock2D(128, 128, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_5",
            nb.ConvBlock2D(128, 256, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_6",
            nb.ConvBlock2D(256, 256, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_7",
            nb.ConvBlock2D(256, 256, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_8",
            nb.ConvBlock2D(256, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_9",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_10",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_11",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_12",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_13",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )

        self.model += nl.Flatten()
        self.model += (
            "DenseBlock_1",
            nb.DenseBlock(512 * 7 * 7, 4096, **asdict(dense_args)),
        )
        self.model += (
            "DenseBlock_2",
            nb.DenseBlock(4096, 4096, **asdict(dense_args)),
        )
        self.model += nl.Dense(
            4096,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_VGGNet_16, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_VGGNet_16, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_VGGNet_16, self).score_nn(X, y, metric, argmax)


class _VGGNet_19(Estimator, NeuralModel, ImageClassifier):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
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
        self.dropout_rate = dropout_rate
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
            [3, 64, 64, 128, 128, *[256] * 4, *[512] * 8],
            [512 * 7 * 7, 4096, 4096, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.set_param_ranges(
            {
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        conv_3x3_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="max",
            random_state=self.random_state,
        )
        conv_3x3_no_pool_args = nb.ConvBlockArgs(
            filter_size=3,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="same",
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_pooling=False,
            random_state=self.random_state,
        )
        dense_args = nb.DenseBlockArgs(
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state,
        )

        self.model += (
            "ConvBlock_1",
            nb.ConvBlock2D(3, 64, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_2",
            nb.ConvBlock2D(64, 64, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_3",
            nb.ConvBlock2D(64, 128, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_4",
            nb.ConvBlock2D(128, 128, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_5",
            nb.ConvBlock2D(128, 256, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_6",
            nb.ConvBlock2D(256, 256, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_7",
            nb.ConvBlock2D(256, 256, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_8",
            nb.ConvBlock2D(256, 256, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_9",
            nb.ConvBlock2D(256, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_10",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_11",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_12",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )

        self.model += (
            "ConvBlock_13",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_14",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_15",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_no_pool_args)),
        )
        self.model += (
            "ConvBlock_16",
            nb.ConvBlock2D(512, 512, **asdict(conv_3x3_pool_args)),
        )

        self.model += nl.Flatten()
        self.model += (
            "DenseBlock_1",
            nb.DenseBlock(512 * 7 * 7, 4096, **asdict(dense_args)),
        )
        self.model += (
            "DenseBlock_2",
            nb.DenseBlock(4096, 4096, **asdict(dense_args)),
        )
        self.model += nl.Dense(
            4096,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)

    @Tensor.force_shape(input_shape)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(_VGGNet_19, self).fit_nn(X, y)

    @override
    @Tensor.force_shape(input_shape)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(_VGGNet_19, self).predict_nn(X, argmax)

    @override
    @Tensor.force_shape(input_shape)
    def score(
        self,
        X: Tensor,
        y: Matrix,
        metric: Evaluator = Accuracy,
        argmax: bool = True,
    ) -> float:
        return super(_VGGNet_19, self).score_nn(X, y, metric, argmax)
