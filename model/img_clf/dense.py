from typing import ClassVar

from luma.interface.util import InitUtil

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl
from luma.neural import functional as F

from ..types import ImageClassifier

Composite = nb.DenseNetBlock.Composite
DenseUnit = nb.DenseNetBlock.DenseUnit
Transition = nb.DenseNetBlock.Transition


class _DenseNet_121(NeuralModel, ImageClassifier):
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

        super(_DenseNet_121, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_DenseNet_121, self).init_model()
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


class _DenseNet_169(NeuralModel, ImageClassifier):
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

        super(_DenseNet_169, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_DenseNet_169, self).init_model()
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


class _DenseNet_201(NeuralModel, ImageClassifier):
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

        super(_DenseNet_201, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_DenseNet_201, self).init_model()
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


class _DenseNet_264(NeuralModel, ImageClassifier):
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

        super(_DenseNet_264, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_DenseNet_264, self).init_model()
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


class _SE_DenseNet_121(NeuralModel, ImageClassifier):
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

        super(_SE_DenseNet_121, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_SE_DenseNet_121, self).init_model()
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


class _SE_DenseNet_169(NeuralModel, ImageClassifier):
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

        super(_SE_DenseNet_169, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_SE_DenseNet_169, self).init_model()
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
