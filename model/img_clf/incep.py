from typing import Self, ClassVar, override
from dataclasses import asdict

from luma.interface.typing import Matrix, Tensor
from luma.interface.util import InitUtil
from luma.preprocessing.encoder import LabelSmoothing

from luma.core.super import Supervised
from luma.neural.base import NeuralModel
from luma.neural import functional as F
from luma.neural import block as nb
from luma.neural import layer as nl

from ..types import ImageClassifier


class _Inception_V1(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
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

        super(_Inception_V1, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Inception_V1, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 64, 64, 192],
            [192, 256, 480, 512, 512, 512, 528, 832, 832],
            [1024, self.out_features],
        ]
        self.feature_shapes_ = [
            self._get_feature_shapes(self.feature_sizes_[0]),
            self._get_feature_shapes(self.feature_sizes_[1]),
        ]

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )
        incep_args = nb.BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=False,
            **base_args,
        )

        self.model.extend(
            nl.Conv2D(3, 64, 7, 2, 3, **base_args),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        self.model.extend(
            nl.Conv2D(64, 64, 1, 1, "valid", **base_args),
            self.activation(),
            nl.Conv2D(64, 192, 3, 1, "valid", **base_args),
            self.activation(),
            nl.Pool2D(3, 2, "max", "same"),
        )

        self.model.extend(
            (
                "Inception_3a",
                nb.IncepBlock.V1(192, 64, 96, 128, 16, 32, 32, **asdict(incep_args)),
            ),
            (
                "Inception_3b",
                nb.IncepBlock.V1(256, 128, 128, 192, 32, 96, 64, **asdict(incep_args)),
            ),
            nl.Pool2D(3, 2, "max", "same"),
            deep_add=False,
        )

        self.model.extend(
            (
                "Inception_4a",
                nb.IncepBlock.V1(480, 192, 96, 208, 16, 48, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4b",
                nb.IncepBlock.V1(512, 160, 112, 224, 24, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4c",
                nb.IncepBlock.V1(512, 128, 128, 256, 24, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4d",
                nb.IncepBlock.V1(512, 112, 144, 288, 32, 64, 64, **asdict(incep_args)),
            ),
            (
                "Inception_4e",
                nb.IncepBlock.V1(
                    528, 256, 160, 320, 32, 128, 128, **asdict(incep_args)
                ),
            ),
            nl.Pool2D(3, 2, "max", "same"),
            deep_add=False,
        )

        self.model.extend(
            (
                "Inception_5a",
                nb.IncepBlock.V1(
                    832, 256, 160, 320, 32, 128, 128, **asdict(incep_args)
                ),
            ),
            (
                "Inception_5b",
                nb.IncepBlock.V1(
                    832, 384, 192, 384, 48, 128, 128, **asdict(incep_args)
                ),
            ),
            nl.GlobalAvgPool2D(),
            nl.Dropout(self.dropout_rate, self.random_state),
            deep_add=False,
        )

        self.model += nl.Flatten()
        self.model += nl.Dense(1024, self.out_features, **base_args)

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _Inception_V2(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
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

        super(_Inception_V2, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Inception_V2, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 32, 32, 64, 64, 80, 192, 288],
            [288, 288, 288, 768],
            [768, 768, 768, 768, 768, 1280],
            [1280, 2048, 2048],
            [2048, self.out_features],
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
        incep_args = nb.BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=False,
            **base_args,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, "valid", **base_args),
            self.activation(),
            nl.Conv2D(32, 32, 3, 1, "valid", **base_args),
            self.activation(),
            nl.Conv2D(32, 64, 3, 1, "same", **base_args),
            self.activation(),
            nl.Pool2D(3, 2, "max", "valid"),
        )

        self.model.extend(
            nl.Conv2D(64, 80, 3, 1, "valid", **base_args),
            self.activation(),
            nl.Conv2D(80, 192, 3, 2, "valid", **base_args),
            self.activation(),
            nl.Conv2D(192, 288, 3, 1, "same", **base_args),
            self.activation(),
        )

        inception_3xA = [
            nb.IncepBlock.V2_TypeA(
                288, 64, 48, 64, 64, (96, 96), 64, **asdict(incep_args)
            )
            for _ in range(3)
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    ["Inception_3a", "Inception_3b", "Inception_3c"], inception_3xA
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx1",
                nb.IncepBlock.V2_Redux(
                    288, 64, 384, 64, (96, 96), **asdict(incep_args)
                ),
            )
        )

        inception_5xB = [
            nb.IncepBlock.V2_TypeB(
                768, 192, 128, 192, 128, (128, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 192, 192, 192, (192, 192), 192, **asdict(incep_args)
            ),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_4a",
                        "Inception_4b",
                        "Inception_4c",
                        "Inception_4d",
                        "Inception_4e",
                    ],
                    inception_5xB,
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx2",
                nb.IncepBlock.V2_Redux(
                    768, 192, 320, 192, (192, 192), **asdict(incep_args)
                ),
            ),
        )

        inception_C_args = [320, 384, (384, 384), 448, 384, (384, 384), 192]
        inception_2xC = [
            nb.IncepBlock.V2_TypeC(1280, *inception_C_args, **asdict(incep_args)),
            nb.IncepBlock.V2_TypeC(2048, *inception_C_args, **asdict(incep_args)),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_5a",
                        "Inception_5b",
                    ],
                    inception_2xC,
                )
            ],
            deep_add=False,
        )

        self.model.add(nl.GlobalAvgPool2D())
        self.model.add(nl.Flatten())
        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.Dense(2048, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)


class _Inception_V3(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.4,
        smoothing: float = 0.1,
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
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_Inception_V3, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Inception_V3, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = [
            [3, 32, 32, 64, 64, 80, 192, 288],
            [288, 288, 288, 768],
            [768, 768, 768, 768, 768, 1280],
            [1280, 2048, 2048],
            [2048, self.out_features],
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
        incep_args = nb.BaseBlockArgs(
            activation=self.activation,
            do_batch_norm=True,
            **base_args,
        )

        self.model.extend(
            nl.Conv2D(3, 32, 3, 2, "valid", **base_args),
            nl.BatchNorm2D(32),
            self.activation(),
            nl.Conv2D(32, 32, 3, 1, "valid", **base_args),
            nl.BatchNorm2D(32),
            self.activation(),
            nl.Conv2D(32, 64, 3, 1, "same", **base_args),
            nl.BatchNorm2D(64),
            self.activation(),
            nl.Pool2D(3, 2, "max", "valid"),
        )

        self.model.extend(
            nl.Conv2D(64, 80, 3, 1, "valid", **base_args),
            nl.BatchNorm2D(80),
            self.activation(),
            nl.Conv2D(80, 192, 3, 2, "valid", **base_args),
            nl.BatchNorm2D(192),
            self.activation(),
            nl.Conv2D(192, 288, 3, 1, "same", **base_args),
            nl.BatchNorm2D(288),
            self.activation(),
        )

        inception_3xA = [
            nb.IncepBlock.V2_TypeA(
                288, 64, 48, 64, 64, (96, 96), 64, **asdict(incep_args)
            )
            for _ in range(3)
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    ["Inception_3a", "Inception_3b", "Inception_3c"], inception_3xA
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx1",
                nb.IncepBlock.V2_Redux(
                    288, 64, 384, 64, (96, 96), **asdict(incep_args)
                ),
            )
        )

        inception_5xB = [
            nb.IncepBlock.V2_TypeB(
                768, 192, 128, 192, 128, (128, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 160, 192, 160, (160, 192), 192, **asdict(incep_args)
            ),
            nb.IncepBlock.V2_TypeB(
                768, 192, 192, 192, 192, (192, 192), 192, **asdict(incep_args)
            ),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_4a",
                        "Inception_4b",
                        "Inception_4c",
                        "Inception_4d",
                        "Inception_4e",
                    ],
                    inception_5xB,
                )
            ],
            deep_add=False,
        )
        self.model.add(
            (
                "Inception_Rx2",
                nb.IncepBlock.V2_Redux(
                    768, 192, 320, 192, (192, 192), **asdict(incep_args)
                ),
            ),
        )

        inception_C_args = [320, 384, (384, 384), 448, 384, (384, 384), 192]
        inception_2xC = [
            nb.IncepBlock.V2_TypeC(1280, *inception_C_args, **asdict(incep_args)),
            nb.IncepBlock.V2_TypeC(2048, *inception_C_args, **asdict(incep_args)),
        ]
        self.model.extend(
            *[
                (name, block)
                for name, block in zip(
                    [
                        "Inception_5a",
                        "Inception_5b",
                    ],
                    inception_2xC,
                )
            ],
            deep_add=False,
        )

        self.model.add(nl.GlobalAvgPool2D())
        self.model.add(nl.Flatten())
        self.model.extend(
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.Dense(2048, self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @override
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_Inception_V3, self).fit_nn(X, y_ls)


class _Inception_V4(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.8,
        smoothing: float = 0.1,
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
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_Inception_V4, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Inception_V4, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        incep_args = nb.BaseBlockArgs(
            activation=self.activation,
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model.add(
            ("Stem", nb.IncepBlock.V4_Stem(**asdict(incep_args))),
        )
        for i in range(1, 5):
            self.model.add(
                (f"Inception_A{i}", nb.IncepBlock.V4_TypeA(**asdict(incep_args))),
            )
        self.model.add(
            (
                "Inception_RA",
                nb.IncepBlock.V4_ReduxA(
                    384, (192, 224, 256, 384), **asdict(incep_args)
                ),
            )
        )
        for i in range(1, 8):
            self.model.add(
                (f"Inception_B{i}", nb.IncepBlock.V4_TypeB(**asdict(incep_args))),
            )
        self.model.add(
            ("Inception_RB", nb.IncepBlock.V4_ReduxB(**asdict(incep_args))),
        )
        for i in range(1, 4):
            self.model.add(
                (f"Inception_C{i}", nb.IncepBlock.V4_TypeC(**asdict(incep_args))),
            )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.Dense(1536, self.out_features),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @override
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_Inception_V4, self).fit_nn(X, y_ls)


class _Inception_ResNet_V1(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.8,
        smoothing: float = 0.1,
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
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_Inception_ResNet_V1, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Inception_ResNet_V1, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        incep_args = nb.BaseBlockArgs(
            activation=self.activation,
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model.add(
            ("Stem", nb.IncepResBlock.V1_Stem(**asdict(incep_args))),
        )
        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_A{i}", nb.IncepResBlock.V1_TypeA(**asdict(incep_args)))
            )
        self.model.add(
            (
                "IncepRes_RA",
                nb.IncepBlock.V4_ReduxA(
                    256, (192, 192, 256, 384), **asdict(incep_args)
                ),
            )
        )
        for i in range(1, 11):
            self.model.add(
                (f"IncepRes_B{i}", nb.IncepResBlock.V1_TypeB(**asdict(incep_args)))
            )
        self.model.add(("IncepRes_RB", nb.IncepResBlock.V1_Redux(**asdict(incep_args))))

        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_C{i}", nb.IncepResBlock.V1_TypeC(**asdict(incep_args)))
            )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.Dense(1792, self.out_features),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @override
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_Inception_ResNet_V1, self).fit_nn(X, y_ls)


class _Inception_ResNet_V2(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.8,
        smoothing: float = 0.1,
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
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_Inception_ResNet_V2, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Inception_ResNet_V2, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        incep_args = nb.BaseBlockArgs(
            activation=self.activation,
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model.add(
            ("Stem", nb.IncepBlock.V4_Stem(**asdict(incep_args))),
        )
        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_A{i}", nb.IncepResBlock.V2_TypeA(**asdict(incep_args))),
            )
        self.model.add(
            (
                "IncepRes_RA",
                nb.IncepBlock.V4_ReduxA(
                    384, (256, 256, 384, 384), **asdict(incep_args)
                ),
            ),
        )

        for i in range(1, 11):
            self.model.add(
                (f"IncepRes_B{i}", nb.IncepResBlock.V2_TypeB(**asdict(incep_args))),
            )
        self.model.add(
            ("IncepRes_RB", nb.IncepResBlock.V2_Redux(**asdict(incep_args))),
        )

        for i in range(1, 6):
            self.model.add(
                (f"IncepRes_C{i}", nb.IncepResBlock.V2_TypeC(**asdict(incep_args))),
            )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.Dense(2272, self.out_features),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @override
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_Inception_ResNet_V2, self).fit_nn(X, y_ls)


class _Xception(NeuralModel, ImageClassifier, Supervised):
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

        super(_Xception, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Xception, self).init_model()
        self.model = nl.Sequential()

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

        self.model.add(("EntryFlow", nb.XceptionBlock.Entry(**base_args)))
        for i in range(1, 9):
            self.model.add(
                (f"MiddleFlow_{i}", nb.XceptionBlock.Middle(**base_args)),
            )

        self.model.extend(
            ("ExitFlow", nb.XceptionBlock.Exit(**base_args)),
            nb.SeparableConv2D(1024, 1536, 3, **base_args),
            nl.BatchNorm2D(1536, self.momentum),
            self.activation(),
            nb.SeparableConv2D(1536, 2048, 3, **base_args),
            nl.BatchNorm2D(2048, self.momentum),
            self.activation(),
            nl.GlobalAvgPool2D(),
            deep_add=False,
        )

        self.model += nl.Flatten()
        self.model += nl.Dense(2048, self.out_features, **base_args)

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)


class _SE_Inception_ResNet_V2(NeuralModel, ImageClassifier, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 1000,
        batch_size: int = 128,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.2,
        smoothing: float = 0.1,
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
        self.smoothing = smoothing
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super(_SE_Inception_ResNet_V2, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_SE_Inception_ResNet_V2, self).init_model()
        self.model = nl.Sequential()

        self.feature_sizes_ = []
        self.feature_shapes_ = [
            self._get_feature_shapes(sizes) for sizes in self.feature_sizes_
        ]

        self.build_model()

    def build_model(self) -> None:
        incep_args = nb.BaseBlockArgs(
            activation=self.activation,
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model += (
            "Stem_SE",
            F.attach_se_block(
                nb.IncepBlock.V4_Stem,
                nb.SEBlock2D,
                asdict(incep_args),
                {"in_channels": 384},
            ),
        )
        for i in range(1, 6):
            self.model += (
                f"IncepRes_A_SE{i}",
                F.attach_se_block(
                    nb.IncepResBlock.V2_TypeA,
                    nb.SEBlock2D,
                    asdict(incep_args),
                    {"in_channels": 384},
                ),
            )
        self.model += (
            "IncepRes_RA_SE",
            F.attach_se_block(
                nb.IncepBlock.V4_ReduxA,
                nb.SEBlock2D,
                {
                    "in_channels": 384,
                    "out_channels_arr": (256, 256, 384, 384),
                    **asdict(incep_args),
                },
                {"in_channels": 1024},
            ),
        )

        for i in range(1, 11):
            self.model += (
                f"IncepRes_B_SE{i}",
                F.attach_se_block(
                    nb.IncepResBlock.V2_TypeB,
                    nb.SEBlock2D,
                    asdict(incep_args),
                    {"in_channels": 1280},
                ),
            )
        self.model += (
            "IncepRes_RB_SE",
            F.attach_se_block(
                nb.IncepResBlock.V2_Redux,
                nb.SEBlock2D,
                asdict(incep_args),
                {"in_channels": 2272},
            ),
        )

        for i in range(1, 6):
            self.model += (
                f"IncepRes_C_SE{i}",
                F.attach_se_block(
                    nb.IncepResBlock.V2_TypeC,
                    nb.SEBlock2D,
                    asdict(incep_args),
                    {"in_channels": 2272},
                ),
            )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dropout(self.dropout_rate, self.random_state),
            nl.Dense(2272, self.out_features),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 299, 299)

    @override
    def fit(self, X: Tensor, y: Matrix) -> Self:
        ls = LabelSmoothing(smoothing=self.smoothing)
        y_ls = ls.fit_transform(y)
        return super(_SE_Inception_ResNet_V2, self).fit_nn(X, y_ls)
