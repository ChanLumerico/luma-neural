from typing import ClassVar
from dataclasses import asdict

from luma.interface.util import InitUtil

from luma.core.super import Supervised
from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl

from ..types import ImageClassifier


class _AlexNet(NeuralModel, ImageClassifier, Supervised):
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

        super(_AlexNet, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_AlexNet, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 96, 256, 384, 384, 256],
            [256 * 6 * 6, 4096, 4096, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.build_model()

    def build_model(self) -> None:
        conv_3x3_no_pool_arg = nb.ConvBlockArgs(
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
            nb.ConvBlock2D(
                3,
                96,
                filter_size=11,
                stride=4,
                activation=self.activation,
                initializer=self.initializer,
                padding="valid",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_1",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_2",
            nb.ConvBlock2D(
                96,
                256,
                filter_size=5,
                stride=1,
                activation=self.activation,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_2",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_3",
            nb.ConvBlock2D(256, 384, **asdict(conv_3x3_no_pool_arg)),
        )
        self.model += (
            "LRN_3",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_4",
            nb.ConvBlock2D(384, 384, **asdict(conv_3x3_no_pool_arg)),
        )
        self.model += (
            "LRN_4",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_5",
            nb.ConvBlock2D(
                384,
                256,
                filter_size=3,
                stride=1,
                activation=self.activation,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_5",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += nl.Flatten()
        self.model += (
            "DenseBlock_1",
            nb.DenseBlock(256 * 6 * 6, 4096, **asdict(dense_args)),
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

    input_shape: ClassVar[tuple] = (-1, 3, 227, 227)


class _ZFNet(NeuralModel, ImageClassifier, Supervised):
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

        super(_ZFNet, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ZFNet, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 96, 256, 384, 384, 256],
            [256 * 6 * 6, 4096, 4096, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.build_model()

    def build_model(self) -> None:
        conv_3x3_no_pool_arg = nb.ConvBlockArgs(
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
            nb.ConvBlock2D(
                3,
                96,
                filter_size=7,
                stride=2,
                activation=self.activation,
                initializer=self.initializer,
                padding="valid",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_1",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_2",
            nb.ConvBlock2D(
                96,
                256,
                filter_size=5,
                stride=2,
                activation=self.activation,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_2",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_3",
            nb.ConvBlock2D(256, 384, **asdict(conv_3x3_no_pool_arg)),
        )
        self.model += (
            "LRN_3",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_4",
            nb.ConvBlock2D(384, 384, **asdict(conv_3x3_no_pool_arg)),
        )
        self.model += (
            "LRN_4",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += (
            "ConvBlock_5",
            nb.ConvBlock2D(
                384,
                256,
                filter_size=3,
                stride=1,
                activation=self.activation,
                initializer=self.initializer,
                padding="same",
                lambda_=self.lambda_,
                do_batch_norm=False,
                pool_filter_size=3,
                pool_stride=2,
                pool_mode="max",
                random_state=self.random_state,
            ),
        )
        self.model += (
            "LRN_5",
            nl.LocalResponseNorm(depth=5),
        )

        self.model += nl.Flatten()
        self.model += (
            "DenseBlock_1",
            nb.DenseBlock(256 * 6 * 6, 4096, **asdict(dense_args)),
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

    input_shape: ClassVar[tuple] = (-1, 3, 227, 227)
