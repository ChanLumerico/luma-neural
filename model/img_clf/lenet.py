from typing import ClassVar

from luma.interface.util import InitUtil

from luma.core.super import Supervised
from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl

from ..types import ImageClassifier


class _LeNet_1(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_LeNet_1, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_LeNet_1, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [1, 4, 8],
            [8 * 4 * 4, self.out_features],
        ]
        self.feature_shapes_ = [
            [(1, 4), (4, 8)],
            [(8 * 4 * 4, self.out_features)],
        ]

        self.build_model()

    def build_model(self) -> None:
        self.model += nb.ConvBlock2D(
            1,
            4,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += nb.ConvBlock2D(
            4,
            8,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += nl.Flatten()
        self.model += nl.Dense(
            8 * 4 * 4,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 1, 28, 28)


class _LeNet_4(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
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

        super(_LeNet_4, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_LeNet_4, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [1, 4, 16],
            [16 * 5 * 5, 120, self.out_features],
        ]
        self.feature_shapes_ = [
            [(1, 4), (4, 16)],
            [(16 * 5 * 5, 120), (120, self.out_features)],
        ]

        self.build_model()

    def build_model(self) -> None:
        self.model += nb.ConvBlock2D(
            1,
            4,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += nb.ConvBlock2D(
            4,
            16,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += nl.Flatten()
        self.model += nb.DenseBlock(
            16 * 5 * 5,
            120,
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_dropout=False,
            random_state=self.random_state,
        )
        self.model += nl.Dense(
            120,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 1, 32, 32)


class _LeNet_5(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Tanh,
        initializer: InitUtil.InitStr = None,
        out_features: int = 10,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
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

        super(_LeNet_5, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_LeNet_5, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [1, 6, 16],
            [16 * 5 * 5, 120, 84, self.out_features],
        ]
        self.feature_shapes_ = [
            [(1, 6), (6, 16)],
            [(16 * 5 * 5, 120), (120, 84), (84, self.out_features)],
        ]

        self.build_model()

    def build_model(self) -> None:
        self.model += nb.ConvBlock2D(
            1,
            6,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )
        self.model += nb.ConvBlock2D(
            6,
            16,
            filter_size=5,
            stride=1,
            activation=self.activation,
            initializer=self.initializer,
            padding="valid",
            lambda_=self.lambda_,
            do_batch_norm=False,
            pool_filter_size=2,
            pool_stride=2,
            pool_mode="avg",
            random_state=self.random_state,
        )

        self.model += nl.Flatten()
        self.model += nb.DenseBlock(
            16 * 5 * 5,
            120,
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_dropout=False,
            random_state=self.random_state,
        )
        self.model += nb.DenseBlock(
            120,
            84,
            activation=self.activation,
            lambda_=self.lambda_,
            do_batch_norm=False,
            do_dropout=False,
            random_state=self.random_state,
        )
        self.model += nl.Dense(
            84,
            self.out_features,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

    input_shape: ClassVar[tuple] = (-1, 1, 32, 32)
