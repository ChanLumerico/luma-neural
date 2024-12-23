from dataclasses import asdict
from typing import ClassVar

from luma.core.super import Supervised
from luma.interface.util import InitUtil

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl
from luma.neural import functional as F

from ..types import ImageClassifier

BasicBlock = nb.ResNetBlock.Basic
Bottleneck = nb.ResNetBlock.Bottleneck
PreActBottle = nb.ResNetBlock.PreActBottleneck
Bottleneck_SE = nb.ResNetBlock.Bottleneck_SE
Bottleneck_SK = nb.ResNetBlock.Bottleneck_SK


class _ResNet_18(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_18, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_18, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 64] * 2,
            [128, 128] * 2,
            [256, 256] * 2,
            [512, 512] * 2,
        ]
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
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, BasicBlock, 2, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, BasicBlock, 2, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, BasicBlock, 2, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, BasicBlock, 2, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * BasicBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ResNet_34(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_34, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_34, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 64] * 3,
            [128, 128] * 4,
            [256, 256] * 6,
            [512, 512] * 3,
        ]
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
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, BasicBlock, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, BasicBlock, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, BasicBlock, 6, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, BasicBlock, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * BasicBlock.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ResNet_50(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_50, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_50, self).init_model()
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

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, Bottleneck, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, Bottleneck, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck, 6, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * Bottleneck.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ResNet_101(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_101, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_101, self).init_model()
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

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, Bottleneck, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, Bottleneck, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck, 23, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * Bottleneck.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ResNet_152(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_152, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_152, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 8,
            [256, 256, 1024] * 36,
            [512, 512, 2048] * 3,
        ]
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
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, Bottleneck, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, Bottleneck, 8, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck, 36, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * Bottleneck.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ResNet_200(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_200, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_200, self).init_model()
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

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, PreActBottle, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, PreActBottle, 24, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, PreActBottle, 36, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, PreActBottle, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(
                512 * PreActBottle.expansion,
                self.out_features,
                **base_args,
            ),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ResNet_269(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_269, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_269, self).init_model()
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

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, PreActBottle, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, PreActBottle, 30, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, PreActBottle, 48, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, PreActBottle, 8, 5, base_args, res_args, stride=2
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
            nl.Dense(
                512 * PreActBottle.expansion,
                self.out_features,
                **base_args,
            ),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ResNet_1001(NeuralModel, ImageClassifier, Supervised):
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

        super(_ResNet_1001, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ResNet_1001, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 33,
            [256, 256, 1024] * 99,
            [512, 512, 2048] * 3,
        ]
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
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, PreActBottle, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, PreActBottle, 33, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, PreActBottle, 99, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, PreActBottle, 8, 5, base_args, res_args, stride=2
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
            nl.Dense(
                512 * PreActBottle.expansion,
                self.out_features,
                **base_args,
            ),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _SE_ResNet_50(NeuralModel, ImageClassifier, Supervised):
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

        super(_SE_ResNet_50, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_SE_ResNet_50, self).init_model()
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

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, Bottleneck_SE, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, Bottleneck_SE, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SE, 6, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SE, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * Bottleneck_SE.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _SE_ResNet_152(NeuralModel, ImageClassifier, Supervised):
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

        super(_SE_ResNet_152, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_SE_ResNet_152, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64],
            [64, 64, 256] * 3,
            [128, 128, 512] * 8,
            [256, 256, 1024] * 36,
            [512, 512, 2048] * 3,
        ]
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
        res_args = asdict(
            nb.BaseBlockArgs(
                activation=self.activation,
                do_batch_norm=True,
                momentum=self.momentum,
                **base_args,
            )
        )

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, Bottleneck_SE, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, Bottleneck_SE, 8, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SE, 36, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SE, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * Bottleneck_SE.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _SK_ResNet_50(NeuralModel, ImageClassifier, Supervised):
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

        super(_SK_ResNet_50, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_SK_ResNet_50, self).init_model()
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

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, Bottleneck_SK, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, Bottleneck_SK, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SK, 6, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SK, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * Bottleneck_SE.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _SK_ResNet_101(NeuralModel, ImageClassifier, Supervised):
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

        super(_SK_ResNet_101, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_SK_ResNet_101, self).init_model()
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

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            nl.BatchNorm2D(64, self.momentum),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )
        self.layer_2, in_channels = F.make_res_layers(
            64, 64, Bottleneck_SK, 3, 2, base_args, res_args
        )
        self.layer_3, in_channels = F.make_res_layers(
            in_channels, 128, Bottleneck_SK, 4, 3, base_args, res_args, stride=2
        )
        self.layer_4, in_channels = F.make_res_layers(
            in_channels, 256, Bottleneck_SK, 23, 4, base_args, res_args, stride=2
        )
        self.layer_5, in_channels = F.make_res_layers(
            in_channels, 512, Bottleneck_SK, 3, 5, base_args, res_args, stride=2
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
            nl.Dense(512 * Bottleneck_SE.expansion, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)
