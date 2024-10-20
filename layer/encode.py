from typing import Tuple
import numpy as np

from luma.interface.typing import Tensor
from luma.neural.base import Layer


__all__ = "_PositionalEncoding"


class _PositionalEncoding(Layer):
    def __init__(self, d_model: int, max_length: int = 5000) -> None:
        super().__init__()
        pe = np.zeros((max_length, d_model))
        position = np.arange(0, max_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(1e4) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        _ = is_train
        N, L, D = X.shape
        self.input_ = X

        self.output_ = X + self.pe[:L, :]
        return self.output_

    def backward(self, d_out: Tensor) -> Tensor:
        dX = d_out
        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape
