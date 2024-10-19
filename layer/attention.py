from typing import Optional, Tuple
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

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, is_train: bool = False
    ) -> Tensor:
        _ = is_train
        self.Q = Q
        self.K = K
        self.V = V

        d_k = Q.shape[-1]
        self.scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        if self.mask is not None:
            self.scores = np.where(self.mask == 0, -np.inf, self.scores)

        scores_max = np.max(self.scores, axis=-1, keepdims=True)
        scores_exp = np.exp(self.scores - scores_max)

        self.attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        self.output_ = np.matmul(self.attention_weights, V)

        return self.output_

    def backward(self, d_out: Tensor) -> Tensor:
        dV = np.matmul(self.attention_weights.transpose(0, 1, 3, 2), d_out)
        d_attention_weights = np.matmul(d_out, self.V.transpose(0, 1, 3, 2))

        sum_d_attention = np.sum(
            d_attention_weights * self.attention_weights, axis=-1, keepdims=True
        )
        d_scores = self.attention_weights * (d_attention_weights - sum_d_attention)

        d_k = self.Q.shape[-1]
        sqrt_dk = np.sqrt(d_k)

        dQ = np.matmul(d_scores, self.K) / sqrt_dk
        dK = np.matmul(d_scores.transpose(0, 1, 3, 2), self.Q) / sqrt_dk

        return dQ, dK, dV

    def out_shape(self, in_shape: tuple[int]) -> tuple[int]:
        return in_shape


class _MultiheadAttention(Layer):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mask: Optional[Tensor] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        rng_ = np.random.RandomState(random_state)
        self.WQ = rng_.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.bQ = np.zeros(d_model)

        self.WK = rng_.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.bK = np.zeros(d_model)

        self.WV = rng_.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.bV = np.zeros(d_model)

        self.W_out = rng_.randn(d_model, d_model) / (1.0 / np.sqrt(d_model))
        self.b_out = np.zeros(d_model)

        self.weights_ = [self.WQ, self.WK, self.WV, self.W_out]
        self.biases_ = [self.bQ, self.bK, self.bV, self.b_out]

        self.attention = _ScaledDotProductAttention(mask=mask)

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        N, L, _ = X.shape

        Q = np.matmul(X, self.WQ) + self.bQ
        K = np.matmul(X, self.WK) + self.bK
        V = np.matmul(X, self.WV) + self.bV

        Q = Q.reshape(N, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(N, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(N, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        attention_output = self.attention.forward(Q, K, V, is_train=is_train)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            N, L, self.n_heads * self.d_v
        )

        output = np.matmul(attention_output, self.W_out) + self.b_out
        return output

    def backward(self, d_out: Tensor) -> Tensor:
        return super().backward(d_out)

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return super().out_shape(in_shape)
