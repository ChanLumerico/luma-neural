from typing import ClassVar

from luma.interface.util import InitUtil

from luma.neural.base import NeuralModel
from luma.neural import block as nb
from luma.neural import layer as nl

from ..types import ImageClassifier


class _ConvNeXt_T(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_T, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_T, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 9, 3]
        ch_arr = [96, 192, 384, 768]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )
        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V1(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_S(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_S, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_S, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 27, 3]
        ch_arr = [96, 192, 384, 768]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )
        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V1(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_B(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_B, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_B, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 27, 3]
        ch_arr = [128, 256, 512, 1024]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )
        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V1(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_L(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_L, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_L, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 27, 3]
        ch_arr = [192, 384, 768, 1536]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )
        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V1(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_XL(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_XL, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_XL, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 27, 3]
        ch_arr = [256, 512, 1024, 2048]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )
        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V1(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_A(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_A, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_A, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [2, 2, 6, 2]
        ch_arr = [40, 80, 160, 320]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_F(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_F, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_F, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [2, 2, 6, 2]
        ch_arr = [48, 96, 192, 384]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_P(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_P, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_P, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [2, 2, 6, 2]
        ch_arr = [64, 128, 256, 512]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_N(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_N, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_N, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [2, 2, 8, 2]
        ch_arr = [80, 160, 320, 640]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_T(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_T, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_T, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 9, 3]
        ch_arr = [96, 192, 384, 768]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_B(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_B, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_B, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 27, 3]
        ch_arr = [128, 256, 512, 1024]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_L(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_L, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_L, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 27, 3]
        ch_arr = [192, 384, 768, 1536]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)


class _ConvNeXt_V2_H(NeuralModel, ImageClassifier):
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

        super(_ConvNeXt_V2_H, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_ConvNeXt_V2_H, self).init_model()
        self.model = nl.Sequential()

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            initializer=self.initializer,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        dep_arr = [3, 3, 27, 3]
        ch_arr = [352, 704, 1408, 2816]
        self.model.extend(
            nl.Conv2D(3, ch_arr[0], 4, 4, "valid", **base_args),
            nl.LayerNorm(),
        )

        for i in range(len(ch_arr)):
            for j in range(dep_arr[i]):
                self.model += (
                    f"ConvNeXt{i + 1}_{j + 1}",
                    nb.ConvNeXtBlock.V2(ch_arr[i], 7, self.activation, **base_args),
                )
            if i < len(ch_arr) - 1:
                self.model.extend(
                    nl.LayerNorm(),
                    nl.Conv2D(ch_arr[i], ch_arr[i + 1], 2, 2, "valid", **base_args),
                )

        self.model.extend(
            nl.GlobalAvgPool2D(),
            nl.LayerNorm(),
            nl.Flatten(),
            nl.Dense(ch_arr[-1], self.out_features, **base_args),
        )

    input_shape: ClassVar[tuple] = (-1, 3, 224, 224)
