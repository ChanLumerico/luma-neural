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
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        is_train: bool = False,
        # How can resolve this multi-input Q, K, V with the Luma-neural system?
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
        WQ = rng_.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        bQ = np.zeros(d_model)

        WK = rng_.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        bK = np.zeros(d_model)

        WV = rng_.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        bV = np.zeros(d_model)

        W_out = rng_.randn(d_model, d_model) / (1.0 / np.sqrt(d_model))
        b_out = np.zeros(d_model)

        self.weights_ = [WQ, WK, WV, W_out]
        self.biases_ = [bQ, bK, bV, b_out]

        self.attention = _ScaledDotProductAttention(mask=mask)

    def forward(self, X: Tensor, is_train: bool = False) -> Tensor:
        self.input_ = X
        N, L, _ = X.shape

        Q = np.matmul(X, self.weights_[0]) + self.biases_[0]
        K = np.matmul(X, self.weights_[1]) + self.biases_[1]
        V = np.matmul(X, self.weights_[2]) + self.biases_[2]

        Q = Q.reshape(N, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(N, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(N, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        att_out = self.attention.forward(Q, K, V, is_train=is_train)
        att_out = att_out.transpose(0, 2, 1, 3).reshape(N, L, self.n_heads * self.d_v)
        self.att_out = att_out

        self.output_ = np.matmul(att_out, self.weights_[3]) + self.biases_[3]
        return self.output_

    def backward(self, d_out: Tensor) -> Tensor:
        N, L, _ = d_out.shape

        d_att_out = np.matmul(d_out, self.weights_[3].T)
        dW_out = np.matmul(
            self.att_out.reshape(N * L, self.d_model).T,
            d_out.reshape(N * L, self.d_model),
        )
        db_out = np.sum(d_out, axis=(0, 1))

        d_att_out = d_att_out.reshape(N, L, self.n_heads, self.d_v)
        d_att_out = d_att_out.transpose(0, 2, 1, 3)

        dQ, dK, dV = self.attention.backward(d_att_out)
        dQ = dQ.transpose(0, 2, 1, 3).reshape(N, L, self.n_heads * self.d_k)
        dK = dK.transpose(0, 2, 1, 3).reshape(N, L, self.n_heads * self.d_k)
        dV = dV.transpose(0, 2, 1, 3).reshape(N, L, self.n_heads * self.d_v)

        dWQ = np.matmul(
            self.input_.reshape(N * L, self.d_model).T,
            dQ.reshape(N * L, self.d_model),
        )
        dbQ = np.sum(dQ, axis=(0, 1))

        dWK = np.matmul(
            self.input_.reshape(N * L, self.d_model).T,
            dK.reshape(N * L, self.d_model),
        )
        dbK = np.sum(dK, axis=(0, 1))

        dWV = np.matmul(
            self.input_.reshape(N * L, self.d_model).T,
            dV.reshape(N * L, self.d_model),
        )
        dbV = np.sum(dV, axis=(0, 1))

        self.dW = [dWQ, dWK, dWV, dW_out]
        self.dB = [dbQ, dbK, dbV, db_out]

        dX_Q = np.matmul(dQ, self.weights_[0].T)
        dX_K = np.matmul(dK, self.weights_[1].T)
        dX_V = np.matmul(dV, self.weights_[2].T)

        dX = dX_Q + dX_K + dX_V
        self.dX = dX
        return self.dX

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape
