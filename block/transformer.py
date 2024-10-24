from typing import Tuple, override

from luma.core.super import Optimizer
from luma.interface.typing import Tensor, TensorLike
from luma.interface.util import InitUtil

from luma.neural import layer as nl
from luma.neural.autoprop import LayerNode, SequentialNode, LayerGraph, MergeMode


class _PositionwiseFeedForward(nl.Sequential):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        basic_args = dict(
            initializer=initializer, lambda_=lambda_, random_state=random_state
        )
        self.set_param_ranges(
            {"in_channels": ("0<,+inf", int), "lambda_": ("0,+inf", None)}
        )
        self.check_param_ranges()

        super(_PositionwiseFeedForward, self).__init__(
            nl.Conv1D(d_model, d_ff, filter_size=1, **basic_args),
            activation(),
            nl.Conv1D(d_ff, d_model, filter_size=1, **basic_args),
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)


class _Encoder(LayerGraph):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        mask: Tensor | None = None,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        dropout_rate: float = 0.1,
        do_buffer: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.mask = mask
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.do_buffer = do_buffer
        self.random_state = random_state

        self.basic_args = dict(
            activation=activation,
            optimizer=optimizer,
            initializer=initializer,
            lambda_=lambda_,
            random_state=random_state,
        )
        self.init_nodes()

        super(_Encoder, self).__init__(
            graph={
                self.rt_: [self.mha_, self.ln_1],
                self.mha_: [self.ln_1],
                self.ln_1: [self.ffn_, self.ln_2],
                self.ffn_: [self.ln_2],
                self.ln_2: [self.buffer_],
                self.buffer_: [self.tm_],
            },
            root=self.rt_,
            term=self.tm_,
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.tm_ = LayerNode(nl.Identity(), name="tm_")

        self.mha_ = SequentialNode(
            nl.MultiHeadAttention(
                self.d_model, self.n_heads, self.mask, self.random_state
            ),
            nl.Dropout(self.dropout_rate, self.random_state),
            name="mha_",
        )
        self.ln_1 = LayerNode(nl.LayerNorm(), name="ln_1")

        self.ffn_ = SequentialNode(
            _PositionwiseFeedForward(self.d_model, self.d_ff, **self.basic_args),
            nl.Dropout(self.dropout_rate, self.random_state),
            name="ffn_",
        )
        self.ln_2 = LayerNode(nl.LayerNorm(), MergeMode.SUM, name="ln_2")
        self.buffer_ = LayerNode(
            nl.Buffer() if self.do_buffer else nl.Identity(), name="buffer_"
        )

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape


class _Decoder(LayerGraph):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        encoder: _Encoder | None = None,
        mask_self: Tensor | None = None,
        mask_enc_dec: Tensor | None = None,
        activation: callable = nl.Activation.ReLU,
        optimizer: Optimizer | None = None,
        initializer: InitUtil.InitStr = None,
        lambda_: float = 0.0,
        dropout_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.mask_self = mask_self
        self.mask_enc_dec = mask_enc_dec
        self.activation = activation
        self.optimizer = optimizer
        self.initializer = initializer
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        if encoder is not None:
            if not hasattr(encoder, "buffer_"):
                raise ValueError("The given encoder has no 'Buffer' layer!")
            self.enc_buf: nl.Buffer = encoder.buffer_.layer
        else:
            self.enc_buf = None

        self.basic_args = dict(
            activation=activation,
            optimizer=optimizer,
            initializer=initializer,
            lambda_=lambda_,
            random_state=random_state,
        )
        self.init_nodes()

        super(_Decoder, self).__init__(
            graph={
                self.rt_: [self.mha_self, self.ln_1],
                self.mha_self: [self.ln_1],
                self.ln_1: [self.mha_enc_dec, self.ln_2],
                self.mha_enc_dec: [self.drop_enc_dec],
                self.drop_enc_dec: [self.ln_2],
                self.ln_2: [self.ffn_, self.ln_3],
                self.ffn_: [self.ln_3],
                self.ln_3: [self.tm_],
            },
            root=self.rt_,
            term=self.tm_,
        )

        if optimizer is not None:
            self.set_optimizer(optimizer)

    def init_nodes(self) -> None:
        self.rt_ = LayerNode(nl.Identity(), name="rt_")
        self.tm_ = LayerNode(nl.Identity(), name="tm_")

        self.mha_self = SequentialNode(
            nl.MultiHeadAttention(
                self.d_model, self.n_heads, self.mask_self, self.random_state
            ),
            nl.Dropout(self.dropout_rate, self.random_state),
            name="mha_self",
        )
        self.ln_1 = LayerNode(nl.LayerNorm(), MergeMode.SUM, name="ln_1")

        self.mha_enc_dec = LayerNode(
            nl.CrossMultiHeadAttention(
                self.d_model, self.n_heads, self.mask_enc_dec, self.random_state
            ),
            name="mha_enc_dec",
        )
        self.drop_enc_dec = LayerNode(
            nl.Dropout(self.dropout_rate, self.random_state), name="drop_enc_dec"
        )
        self.ln_2 = LayerNode(nl.LayerNorm(), MergeMode.SUM, name="ln_2")

        self.ffn_ = SequentialNode(
            _PositionwiseFeedForward(self.d_model, self.d_ff, **self.basic_args),
            nl.Dropout(self.dropout_rate, self.random_state),
            name="ffn_",
        )
        self.ln_3 = LayerNode(nl.LayerNorm(), MergeMode.SUM, name="ln_3")

    @override
    def forward(self, X: TensorLike, is_train: bool = False) -> TensorLike:
        if self.enc_buf is not None:
            self.mha_enc_dec.layer.extern_key_val = self.enc_buf.output_

        return super().forward(X, is_train)

    @override
    def backward(self, d_out: TensorLike) -> TensorLike:
        d_out = super().backward(d_out)

        if self.enc_buf is not None:
            xmha_layer: nl.CrossMultiHeadAttention = self.mha_enc_dec.layer
            dX_K, dX_V = xmha_layer.dX_K, xmha_layer.dX_V

            self.enc_buf.add_back_buffer(dX_K)
            self.enc_buf.add_back_buffer(dX_V)

        return d_out

    def out_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        return in_shape
