from typing import Tuple, Type
import numpy as np

from luma.interface.typing import ClassType, Tensor, TensorLike
from luma.neural.base import Layer


__all__ = "_Activation"


@ClassType.non_instantiable()
class _Activation:

    @classmethod
    def _out_shape(cls, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape

    class Linear(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            return X

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class ReLU(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            return np.maximum(0, X)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out.copy()
            self.dX[self.input_ <= 0] = 0
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class ReLU6(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
            _ = is_train
            self.input_ = X
            return np.minimum(np.maximum(0, X), 6)

        def backward(self, d_out: TensorLike) -> TensorLike:
            self.dX = d_out.copy()
            self.dX[(self.input_ <= 0) | (self.input_ > 6)] = 0
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class Sigmoid(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.output_ = 1 / (1 + np.exp(-X))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * self.output_ * (1 - self.output_)
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class Tanh(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.output_ = np.tanh(X)
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (1 - np.square(self.output_))
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class LeakyReLU(Layer):
        def __init__(self, alpha: float = 0.01) -> None:
            super().__init__()
            self.alpha = alpha

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            return np.where(X > 0, X, X * self.alpha)

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * np.where(self.input_ > 0, 1, self.alpha)
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class Softmax(Layer):
        def __init__(self, dim: int = -1) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X

            shifted_X = X - np.max(X, axis=self.dim, keepdims=True)
            e_X = np.exp(shifted_X)

            self.output_ = e_X / np.sum(e_X, axis=self.dim, keepdims=True)
            return self.output_

        def backward(self, d_out: TensorLike) -> TensorLike:
            sum_d_out_y = np.sum(d_out * self.output_, axis=self.dim, keepdims=True)

            self.dX = self.output_ * (d_out - sum_d_out_y)
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class ELU(Layer):
        def __init__(self, alpha: float = 1.0) -> None:
            super().__init__()
            self.alpha = alpha

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            self.output_ = np.where(X > 0, X, self.alpha * (np.exp(X) - 1))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * np.where(
                self.input_ > 0,
                1,
                self.output_ + self.alpha,
            )
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class SELU(Layer):
        def __init__(
            self,
            lambda_: float = 1.0507,
            alpha: float = 1.67326,
        ) -> None:
            super().__init__()
            self.lambda_ = lambda_
            self.alpha = alpha

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            self.output_ = self.lambda_ * np.where(
                X > 0, X, self.alpha * (np.exp(X) - 1)
            )
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = (
                d_out
                * self.lambda_
                * np.where(
                    self.input_ > 0,
                    1,
                    self.alpha * np.exp(self.input_),
                )
            )
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class Softplus(Layer):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.output_ = np.log1p(np.exp(X))
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (1 - 1 / (1 + np.exp(self.output_)))
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class Swish(Layer):
        def __init__(self, beta: float = 1.0) -> None:
            super().__init__()
            self.beta = beta

        def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
            _ = is_train
            self.input_ = X
            self.sigmoid = 1 / (1 + np.exp(-self.beta * X))

            self.output_ = X * self.sigmoid
            return self.output_

        def backward(self, d_out: Tensor) -> Tensor:
            self.dX = d_out * (
                self.sigmoid
                + self.beta * self.input_ * self.sigmoid * (1 - self.sigmoid)
            )
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class HardSwish(Layer):
        def __init__(self) -> None:
            super().__init__()
            self.relu6 = _Activation.ReLU6()

        def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
            _ = is_train
            self.input_ = X
            relu6_output = self.relu6.forward(X + 3)

            self.output_ = X * (relu6_output / 6)
            return self.output_

        def backward(self, d_out: TensorLike) -> TensorLike:
            relu6_grad = self.relu6.backward(d_out)
            self.dX = d_out * (
                (self.relu6.forward(self.input_ + 3) / 6)
                + (self.input_ * relu6_grad / 6)
            )
            return self.dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)

    class GELU(Layer):
        def __init__(self) -> None:
            super().__init__()
            self.a = np.sqrt(2 / np.pi)
            self.b = 0.044715

            self.input_ = None
            self.tanh_z = None
            self.sech_z_squared = None
            self.z = None

        def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
            _ = is_train
            self.input_ = X

            self.z = self.a * X + self.b * (X**3)
            self.tanh_z = np.tanh(self.z)

            output = 0.5 * X * (1 + self.tanh_z)
            return output

        def backward(self, d_out: TensorLike) -> TensorLike:
            self.sech_z_squared = 1 - self.tanh_z**2
            dz_dx = self.a + 3 * self.b * (self.input_**2)

            dGELU = (
                0.5 * (1 + self.tanh_z)
                + 0.5 * self.input_ * self.sech_z_squared * dz_dx
            )

            dX = d_out * dGELU
            return dX

        def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
            return _Activation._out_shape(in_shape)
