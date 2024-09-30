from typing import Optional, Tuple
import numpy as np

from luma.core.super import Optimizer

from luma.interface.typing import TensorLike, Tensor, Matrix
from luma.interface.util import InitUtil
from luma.interface.exception import NotFittedError

from luma.neural.base import Layer


__all__ = (
    "_Flatten",
    "_Reshape",
    "_Transpose",
    "_Dense",
)


class _Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: Tensor, is_train: bool = False) -> Matrix:
        _ = is_train
        self.input_ = X
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out: Matrix) -> Tensor:
        dX = d_out.reshape(self.input_.shape)
        return dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, *shape = in_shape
        return (batch_size, int(np.prod(shape)))


class _Reshape(Layer):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.target_shape = shape
        self.input_shape = None

        if shape.count(-1) > len(shape):
            raise ValueError("Invalid number of -1s in shape.")

    def _compute_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        target_shape = list(self.target_shape)
        num_neg_ones = target_shape.count(-1)

        if num_neg_ones > len(in_shape):
            raise ValueError("Too many -1s in target shape.")

        input_ptr = 0
        for i, dim in enumerate(target_shape):
            if dim == -1:
                if input_ptr >= len(in_shape):
                    raise ValueError("Not enough input axes to replace -1.")

                target_shape[i] = in_shape[input_ptr]
                input_ptr += 1

        if np.prod(target_shape) != np.prod(in_shape):
            raise ValueError(
                f"Cannot reshape array of size {np.prod(in_shape)}"
                + f" into shape {tuple(target_shape)}"
            )

        return tuple(target_shape)

    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        _ = is_train
        self.input_shape = X.shape

        target_shape = self._compute_shape(self.input_shape)
        return X.reshape(target_shape)

    def backward(self, d_out: TensorLike) -> TensorLike:
        if self.input_shape is None:
            raise ValueError("Must perform forward pass before backward.")

        if np.prod(d_out.shape) != np.prod(self.input_shape):
            raise ValueError(
                f"Cannot reshape gradient array of size {np.prod(d_out.shape)} "
                + f" into shape {self.input_shape}"
            )

        return d_out.reshape(self.input_shape)

    def out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return self._compute_shape(in_shape)


class _Transpose(Layer):
    def __init__(self, axes: Optional[tuple[int]] = None):
        super().__init__()
        self.axes = axes
        self.inverse_axes = None

    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        _ = is_train
        if self.axes is None:
            self.axes = tuple(reversed(range(X.ndim)))
        else:
            if sorted(self.axes) != list(range(X.ndim)):
                raise ValueError(f"Invalid permutation of axes: {self.axes}")

        self.inverse_axes = np.argsort(self.axes)
        return X.transpose(self.axes)

    def backward(self, d_out: TensorLike) -> TensorLike:
        if self.inverse_axes is None:
            raise NotFittedError("Must perform forward pass before backward!")

        return d_out.transpose(self.inverse_axes)

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        if self.axes is None:
            return tuple(reversed(in_shape))
        else:
            return tuple(in_shape[axis] for axis in self.axes)


class _Dense(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer: InitUtil.InitStr = None,
        optimizer: Optimizer = None,
        lambda_: float = 0.0,
        random_state: int = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initializer = initializer
        self.optimizer = optimizer
        self.lambda_ = lambda_
        self.random_state = random_state
        self.rs_ = np.random.RandomState(self.random_state)

        self.init_params(
            w_shape=(self.in_features, self.out_features),
            b_shape=(1, self.out_features),
        )
        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "lambda_": ("0,+inf", None),
            }
        )
        self.check_param_ranges()

    def forward(self, X: Matrix, is_train: bool = False) -> Matrix:
        _ = is_train
        self.input_ = X
        return np.dot(X, self.weights_) + self.biases_

    def backward(self, d_out: Matrix) -> Matrix:
        X = self.input_

        self.dX = np.dot(d_out, self.weights_.T)
        self.dW = np.dot(X.T, d_out)
        self.dW += 2 * self.lambda_ * self.weights_
        self.dB = np.sum(d_out, axis=0, keepdims=True)

        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        batch_size, _ = in_shape
        return (batch_size, self.out_features)
