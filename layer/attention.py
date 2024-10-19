from typing import Optional
import numpy as np

from luma.interface.typing import Tensor
from luma.neural.base import Layer


class _ScaledDotProductAttention(Layer):
    def __init__(self, mask: Optional[Tensor] = None) -> None:
        super().__init__()
        self.mask = mask

        self.Q = None
        self.K = None
        self.V = None

        self.scores = None
        self.attention_weights = None

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, is_train: bool = False
    ) -> Tensor:
        _ = is_train
        self.Q = Q  # (N, H, Lq, dk)
        self.K = K  # (N, H, Lk, dk)
        self.V = V  # (N, H, Lv, dv)
        self.input_ = [Q, K, V]  # Might be optional

        d_k = Q.shape[-1]
        self.scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        if self.mask is not None:
            self.scores = np.where(self.mask == 0, -1e9, self.scores)

        scores_max = np.max(self.scores, axis=-1, keepdims=True)
        scores_exp = np.exp(self.scores - scores_max)

        self.attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        self.output_ = np.matmul(self.attention_weights, V)

        return self.output_

    def backward(self, d_out: Tensor) -> Tensor:
        return super().backward(d_out)

    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        return super().out_shape(in_shape)
