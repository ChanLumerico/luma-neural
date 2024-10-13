from typing import override, ClassVar

from luma.interface.typing import TensorLike
from luma.interface.util import InitUtil
from luma.preprocessing.image import Resize

from luma.core.super import Supervised
from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl
from luma.neural import functional as F

from ..types import ImageClassifier

MBConv = nb.EfficientBlock.MBConv
FusedMBConv = nb.EfficientBlock.FusedMBConv

b0_config = [
    [16, 1, 1, 1, 3],
    [24, 2, 6, 2, 3],
    [40, 2, 6, 2, 5],
    [80, 3, 6, 2, 3],
    [112, 3, 6, 1, 5],
    [192, 4, 6, 2, 5],
    [320, 1, 6, 1, 3],
]

multipliers = [
    [1.0, 1.0],
    [1.1, 1.2],
    [1.21, 1.44],
    [1.33, 1.72],
    [1.46, 2.07],
    [1.61, 2.48],
    [1.77, 2.99],
    [1.95, 3.58],
]

input_sizes = [224, 240, 260, 300, 380, 456, 528, 600]


class _EfficientNet_B0(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B0, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B0, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=0,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[0], input_sizes[0])


class _EfficientNet_B1(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B1, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B1, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=1,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[1], input_sizes[1])


class _EfficientNet_B2(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B2, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B2, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=2,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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

        dense_in_features = int(round(1280 * multipliers[2][0]))
        self.model.extend(
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[2], input_sizes[2])


class _EfficientNet_B3(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B3, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B3, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=3,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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

        dense_in_features = int(round(1280 * multipliers[3][0]))
        self.model.extend(
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[3], input_sizes[3])


class _EfficientNet_B4(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B4, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B4, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=4,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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

        dense_in_features = int(round(1280 * multipliers[4][0]))
        self.model.extend(
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[4], input_sizes[4])


class _EfficientNet_B5(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B5, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B5, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=5,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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

        dense_in_features = int(round(1280 * multipliers[5][0]))
        self.model.extend(
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[5], input_sizes[5])


class _EfficientNet_B6(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B6, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B6, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=6,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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

        dense_in_features = int(round(1280 * multipliers[6][0]))
        self.model.extend(
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[6], input_sizes[6])


class _EfficientNet_B7(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
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
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
        self.momentum = momentum
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_EfficientNet_B7, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_B7, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = F.get_efficient_net_mbconv_config(
            b0_config,
            multipliers,
            n=7,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
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

        dense_in_features = int(round(1280 * multipliers[7][0]))
        self.model.extend(
            nl.Conv2D(in_, dense_in_features, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(dense_in_features, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(dense_in_features, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, input_sizes[7], input_sizes[7])


class _EfficientNet_V2_S(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        dropout_rate: float = 0.1,
        progressive_learning: bool = True,
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
        self.dropout_rate = dropout_rate
        self.progressive_learning = progressive_learning
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False
        self._cur_stage = -1

        super(_EfficientNet_V2_S, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_V2_S, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = [
            [2, 24, 1, 1, True],
            [4, 48, 4, 2, True],
            [4, 64, 4, 2, True],
            [6, 128, 4, 2, False],
            [9, 160, 6, 1, False],
            [15, 256, 6, 2, False],
        ]

        self.model.extend(
            nl.Conv2D(3, 24, 3, 2, **base_args),
            nl.BatchNorm2D(24, self.momentum),
            self.activation(),
        )

        in_ = 24
        for i, (n, out, exp, s, is_fused) in enumerate(mbconv_config):
            block = FusedMBConv if is_fused else MBConv
            for j in range(n):
                s_ = s if j == 0 else 1
                self.model += (
                    f"{block.__name__}{i + 1}_{j + 1}",
                    block(in_, out, 3, s_, exp, 4, **base_args),
                )
                in_ = out

        self.model.extend(
            nl.Conv2D(in_, 1280, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(1280, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(1280, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 384, 384)

    @override
    def train(self, X: TensorLike, y: TensorLike, epoch: int) -> list[float]:
        new_stage = epoch // (self.n_epochs // 4)
        if self._cur_stage != new_stage and self.progressive_learning:
            self._cur_stage = new_stage
            new_res, new_drop_rate = self.update_size_dropout_rate(self._cur_stage)

            X = Resize((new_res, new_res)).fit_transform(X)

            drop_layer = self.model.layers[-4]
            if isinstance(drop_layer, nl.Dropout):
                drop_layer.dropout_rate = new_drop_rate

        return super(_EfficientNet_V2_S, self).train(X, y, epoch)

    def update_size_dropout_rate(self, stage: int) -> tuple[int, float]:
        res_arr = [128, 160, 192, 224]
        drop_rate_arr = [0.1, 0.2, 0.3, 0.4]

        assert stage < 4
        return res_arr[stage], drop_rate_arr[stage]


class _EfficientNet_V2_M(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        dropout_rate: float = 0.1,
        progressive_learning: bool = True,
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
        self.dropout_rate = dropout_rate
        self.progressive_learning = progressive_learning
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False
        self._cur_stage = -1

        super(_EfficientNet_V2_M, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_V2_M, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = [
            [5, 48, 4, 2, True],
            [5, 80, 4, 2, True],
            [7, 160, 4, 2, False],
            [14, 176, 6, 1, False],
            [18, 304, 6, 2, False],
            [5, 512, 6, 1, False],
        ]

        self.model.extend(
            nl.Conv2D(3, 24, 3, 2, **base_args),
            nl.BatchNorm2D(24, self.momentum),
            self.activation(),
        )

        in_ = 24
        for i, (n, out, exp, s, is_fused) in enumerate(mbconv_config):
            block = FusedMBConv if is_fused else MBConv
            for j in range(n):
                s_ = s if j == 0 else 1
                self.model += (
                    f"{block.__name__}{i + 1}_{j + 1}",
                    block(in_, out, 3, s_, exp, 4, **base_args),
                )
                in_ = out

        self.model.extend(
            nl.Conv2D(in_, 1280, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(1280, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(1280, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 480, 480)

    @override
    def train(self, X: TensorLike, y: TensorLike, epoch: int) -> list[float]:
        new_stage = epoch // (self.n_epochs // 4)
        if self._cur_stage != new_stage and self.progressive_learning:
            self._cur_stage = new_stage
            new_res, new_drop_rate = self.update_size_dropout_rate(self._cur_stage)

            X = Resize((new_res, new_res)).fit_transform(X)

            drop_layer = self.model.layers[-4]
            if isinstance(drop_layer, nl.Dropout):
                drop_layer.dropout_rate = new_drop_rate

        return super(_EfficientNet_V2_M, self).train(X, y, epoch)

    def update_size_dropout_rate(self, stage: int) -> tuple[int, float]:
        res_arr = [160, 192, 224, 256]
        drop_rate_arr = [0.1, 0.2, 0.3, 0.4]

        assert stage < 4
        return res_arr[stage], drop_rate_arr[stage]


class _EfficientNet_V2_L(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        dropout_rate: float = 0.1,
        progressive_learning: bool = True,
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
        self.dropout_rate = dropout_rate
        self.progressive_learning = progressive_learning
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False
        self._cur_stage = -1

        super(_EfficientNet_V2_L, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_V2_L, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = [
            [7, 64, 4, 2, True],
            [7, 96, 4, 2, True],
            [10, 192, 4, 2, False],
            [25, 224, 6, 1, False],
            [25, 384, 6, 2, False],
            [7, 640, 6, 1, False],
        ]

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
        )

        in_ = 32
        for i, (n, out, exp, s, is_fused) in enumerate(mbconv_config):
            block = FusedMBConv if is_fused else MBConv
            for j in range(n):
                s_ = s if j == 0 else 1
                self.model += (
                    f"{block.__name__}{i + 1}_{j + 1}",
                    block(in_, out, 3, s_, exp, 4, **base_args),
                )
                in_ = out

        self.model.extend(
            nl.Conv2D(in_, 1280, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(1280, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(1280, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 480, 480)

    @override
    def train(self, X: TensorLike, y: TensorLike, epoch: int) -> list[float]:
        new_stage = epoch // (self.n_epochs // 4)
        if self._cur_stage != new_stage and self.progressive_learning:
            self._cur_stage = new_stage
            new_res, new_drop_rate = self.update_size_dropout_rate(self._cur_stage)

            X = Resize((new_res, new_res)).fit_transform(X)

            drop_layer = self.model.layers[-4]
            if isinstance(drop_layer, nl.Dropout):
                drop_layer.dropout_rate = new_drop_rate

        return super(_EfficientNet_V2_L, self).train(X, y, epoch)

    def update_size_dropout_rate(self, stage: int) -> tuple[int, float]:
        res_arr = [192, 224, 256, 320]
        drop_rate_arr = [0.1, 0.2, 0.3, 0.4]

        assert stage < 4
        return res_arr[stage], drop_rate_arr[stage]


class _EfficientNet_V2_XL(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.Swish,
        initializer: InitUtil.InitStr = None,
        out_features: int = 21843,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        momentum: float = 0.9,
        dropout_rate: float = 0.1,
        progressive_learning: bool = True,
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
        self.dropout_rate = dropout_rate
        self.progressive_learning = progressive_learning
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False
        self._cur_stage = -1

        super(_EfficientNet_V2_XL, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_EfficientNet_V2_XL, self).init_model()
        self.model: nl.Sequential = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        mbconv_config = [
            [8, 64, 4, 2, True],
            [8, 96, 4, 2, True],
            [16, 192, 4, 2, False],
            [24, 256, 6, 1, False],
            [32, 512, 6, 2, False],
            [8, 640, 6, 1, False],
        ]

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, **base_args),
            nl.BatchNorm2D(32, self.momentum),
            self.activation(),
        )

        in_ = 32
        for i, (n, out, exp, s, is_fused) in enumerate(mbconv_config):
            block = FusedMBConv if is_fused else MBConv
            for j in range(n):
                s_ = s if j == 0 else 1
                self.model += (
                    f"{block.__name__}{i + 1}_{j + 1}",
                    block(in_, out, 3, s_, exp, 4, **base_args),
                )
                in_ = out

        self.model.extend(
            nl.Conv2D(in_, 1280, 1, 1, "valid", **base_args),
            nl.BatchNorm2D(1280, self.momentum),
            self.activation(),
        )
        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(1280, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 480, 480)

    @override
    def train(self, X: TensorLike, y: TensorLike, epoch: int) -> list[float]:
        new_stage = epoch // (self.n_epochs // 4)
        if self._cur_stage != new_stage and self.progressive_learning:
            self._cur_stage = new_stage
            new_res, new_drop_rate = self.update_size_dropout_rate(self._cur_stage)

            X = Resize((new_res, new_res)).fit_transform(X)

            drop_layer = self.model.layers[-4]
            if isinstance(drop_layer, nl.Dropout):
                drop_layer.dropout_rate = new_drop_rate

        return super(_EfficientNet_V2_XL, self).train(X, y, epoch)

    def update_size_dropout_rate(self, stage: int) -> tuple[int, float]:
        res_arr = [192, 224, 256, 320]
        drop_rate_arr = [0.1, 0.2, 0.3, 0.4]

        assert stage < 4
        return res_arr[stage], drop_rate_arr[stage]
