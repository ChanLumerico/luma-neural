from typing import Tuple
import numpy as np

from luma.interface.typing import Tensor
from luma.neural.base import Layer


__all__ = (
    "_Dropout",
    "_Dropout1D",
    "_Dropout2D",
    "_Dropout3D",
    "_DropBlock1D",
    "_DropBlock2D",
    "_DropBlock3D",
)


class _Dropout(Layer):
    def __init__(
        self,
        dropout_rate: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        super(_Dropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            self.mask_ = self.rs_.rand(*X.shape) < (1 - self.dropout_rate)
            return X * self.mask_ / (1 - self.dropout_rate)
        return X

    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.dropout_rate)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _Dropout1D(Layer):
    def __init__(
        self,
        dropout_rate: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    @Tensor.force_dim(3)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            self.mask_ = self.rs_.rand(*X.shape[:2], 1) < (1 - self.dropout_rate)
            return X * self.mask_ / (1 - self.dropout_rate)
        return X

    @Tensor.force_dim(3)
    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.dropout_rate)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _Dropout2D(Layer):
    def __init__(
        self,
        dropout_rate: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            self.mask_ = self.rs_.rand(*X.shape[:2], 1, 1) < (1 - self.dropout_rate)
            return X * self.mask_ / (1 - self.dropout_rate)
        return X

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.dropout_rate)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _Dropout3D(Layer):
    def __init__(
        self,
        dropout_rate: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges({"dropout_rate": ("0,1", None)})
        self.check_param_ranges()

    @Tensor.force_dim(5)
    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        if is_train:
            self.mask_ = self.rs_.rand(*X.shape[:2], 1, 1, 1) < (1 - self.dropout_rate)
            return X * self.mask_ / (1 - self.dropout_rate)
        return X

    @Tensor.force_dim(5)
    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.dropout_rate)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _DropBlock1D(Layer):
    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.random_state = random_state

        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges(
            {"drop_prob": ("0,1", None), "block_size": ("0<,+inf", int)}
        )
        self.check_param_ranges()

    @Tensor.force_dim(3)
    def forward(self, X: Tensor, is_train=False) -> Tensor:
        if not is_train or self.drop_prob == 0.0:
            return X
        gamma = self._compute_gamma(X.shape)
        mask = self._generate_mask(X.shape, gamma)

        self.mask_ = mask
        return X * mask / (1 - self.drop_prob)

    @Tensor.force_dim(3)
    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.drop_prob)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape

    def _compute_gamma(self, shape: tuple[int], eps: float = 1e-7) -> float:
        _, _, w = shape
        return (
            self.drop_prob
            * w
            / (self.block_size**2)
            / ((w - self.block_size + 1) + eps)
        )

    def _generate_mask(self, shape: tuple[int], gamma: float) -> Tensor:
        mask = (self.rs_.rand(*shape) < gamma).astype(np.float64)

        pad = self.block_size // 2
        padded_mask = np.pad(
            mask,
            ((0, 0), (0, 0), (pad, pad)),
            mode="constant",
            constant_values=0,
        )
        block_mask = np.zeros_like(mask)

        _, _, w = shape
        for i in range(self.block_size):
            block_mask = np.maximum(block_mask, padded_mask[:, :, i : i + w])

        block_mask = 1 - block_mask
        return block_mask


class _DropBlock2D(Layer):
    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.random_state = random_state

        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges(
            {"drop_prob": ("0,1", None), "block_size": ("0<,+inf", int)}
        )
        self.check_param_ranges()

    @Tensor.force_dim(4)
    def forward(self, X: Tensor, is_train=False) -> Tensor:
        if not is_train or self.drop_prob == 0.0:
            return X
        gamma = self._compute_gamma(X.shape)
        mask = self._generate_mask(X.shape, gamma)

        self.mask_ = mask
        return X * mask / (1 - self.drop_prob)

    @Tensor.force_dim(4)
    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.drop_prob)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape

    def _compute_gamma(self, shape: tuple[int], eps: float = 1e-7) -> float:
        _, _, h, w = shape
        return (
            self.drop_prob
            * (h * w)
            / (self.block_size**2)
            / ((h - self.block_size + 1) * (w - self.block_size + 1) + eps)
        )

    def _generate_mask(self, shape: tuple[int], gamma: float) -> Tensor:
        mask = (self.rs_.rand(*shape) < gamma).astype(np.float64)

        pad = self.block_size // 2
        padded_mask = np.pad(
            mask,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0,
        )
        block_mask = np.zeros_like(mask)

        _, _, h, w = shape
        for i in range(self.block_size):
            for j in range(self.block_size):
                block_mask = np.maximum(
                    block_mask, padded_mask[:, :, i : i + h, j : j + w]
                )

        block_mask = 1 - block_mask
        return block_mask


class _DropBlock3D(Layer):
    def __init__(
        self,
        drop_prob: float = 0.1,
        block_size: int = 7,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.random_state = random_state

        self.rs_ = np.random.RandomState(self.random_state)
        self.mask_ = None

        self.set_param_ranges(
            {"drop_prob": ("0,1", None), "block_size": ("0<,+inf", int)}
        )
        self.check_param_ranges()

    @Tensor.force_dim(5)
    def forward(self, X: Tensor, is_train=False) -> Tensor:
        if not is_train or self.drop_prob == 0.0:
            return X
        gamma = self._compute_gamma(X.shape)
        mask = self._generate_mask(X.shape, gamma)

        self.mask_ = mask
        return X * mask / (1 - self.drop_prob)

    @Tensor.force_dim(5)
    def backward(self, d_out: Tensor) -> Tensor:
        if self.mask_ is not None:
            return d_out * self.mask_ / (1 - self.drop_prob)
        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape

    def _compute_gamma(self, shape: tuple[int], eps: float = 1e-7) -> float:
        _, _, d, h, w = shape
        return (
            self.drop_prob
            * (d * h * w)
            / (self.block_size**3)
            / (
                (d - self.block_size + 1)
                * (h - self.block_size + 1)
                * (w - self.block_size + 1)
                + eps
            )
        )

    def _generate_mask(self, shape: tuple[int], gamma: float) -> Tensor:
        mask = (self.rs_.rand(*shape) < gamma).astype(np.float64)

        pad = self.block_size // 2
        padded_mask = np.pad(
            mask,
            ((0, 0), (0, 0), (pad, pad), (pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0,
        )
        block_mask = np.zeros_like(mask)

        _, _, d, h, w = shape
        for i in range(self.block_size):
            for j in range(self.block_size):
                for k in range(self.block_size):
                    block_mask = np.maximum(
                        block_mask, padded_mask[:, :, i : i + d, j : j + h, k : k + w]
                    )

        block_mask = 1 - block_mask
        return block_mask
