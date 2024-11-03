from typing import override

from luma.core.super import Supervised
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural.base import NeuralModel
from luma.neural import layer as nl
from luma.neural import block as nb
from luma.neural import functional as F

from ..types import SequenceToSequence


class _Transformer_Base(NeuralModel, SequenceToSequence, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 37000,
        batch_size: int = 64,
        n_epochs: int = 10,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.1,
        early_stopping: bool = False,
        patience: int = 3,
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

        super(_Transformer_Base, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Transformer_Base, self).init_model()
        self.model = nl.Sequential()

        self.padding_mask_func = F.generate_padding_mask
        self.look_ahead_mask_func = F.generate_look_ahead_mask

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            activation=self.activation,
            initializer=self.initializer,
            dropout_rate=self.dropout_rate,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.encoder = nb.TransformerBlock.EncoderStack(
            n_encoders=6,
            d_model=512,
            d_ff=2048,
            n_heads=8,
            mask_func=self.padding_mask_func,
            **base_args
        )
        self.decoder = nb.TransformerBlock.DecoderStack(
            n_decoders=6,
            d_model=512,
            d_ff=2048,
            n_heads=8,
            encoder=self.encoder[-1][1],
            mask_self_func=self.look_ahead_mask_func,
            mask_cross_func=self.padding_mask_func,
            **base_args
        )
        self.lin_softmax = nl.Sequential(
            nl.DenseND(
                in_features=512,
                out_features=self.out_features,
                axis=-1,
                initializer=self.initializer,
                lambda_=self.lambda_,
                random_state=self.random_state,
            ),
            nl.Activation.Softmax(dim=-1),
        )

        self.model.extend(self.encoder, self.decoder, self.lin_softmax)

    @override
    def forward(self, X: Tensor, y: Tensor, is_train: bool = False) -> Tensor:
        _ = self.encoder(X, is_train)
        out = self.decoder(y, is_train)
        out = self.lin_softmax(out, is_train)
        return out

    @override
    def backward(self, d_out: Tensor) -> Tensor:
        d_out = self.lin_softmax.backward(d_out)
        _ = self.decoder.backward(d_out)
        d_out = self.encoder.backward(d_out)
        return d_out


class _Transformer_Big(NeuralModel, SequenceToSequence, Supervised):
    def __init__(
        self,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 37000,
        batch_size: int = 64,
        n_epochs: int = 10,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.1,
        early_stopping: bool = False,
        patience: int = 3,
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

        super(_Transformer_Big, self).__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super(_Transformer_Big, self).init_model()
        self.model = nl.Sequential()

        self.padding_mask_func = F.generate_padding_mask
        self.look_ahead_mask_func = F.generate_look_ahead_mask

        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            activation=self.activation,
            initializer=self.initializer,
            dropout_rate=self.dropout_rate,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.encoder = nb.TransformerBlock.EncoderStack(
            n_encoders=6,
            d_model=1024,
            d_ff=4096,
            n_heads=16,
            mask_func=self.padding_mask_func,
            **base_args
        )
        self.decoder = nb.TransformerBlock.DecoderStack(
            n_decoders=6,
            d_model=1024,
            d_ff=4096,
            n_heads=16,
            encoder=self.encoder[-1][1],
            mask_self_func=self.look_ahead_mask_func,
            mask_cross_func=self.padding_mask_func,
            **base_args
        )
        self.lin_softmax = nl.Sequential(
            nl.DenseND(
                in_features=1024,
                out_features=self.out_features,
                axis=-1,
                initializer=self.initializer,
                lambda_=self.lambda_,
                random_state=self.random_state,
            ),
            nl.Activation.Softmax(dim=-1),
        )

        self.model.extend(self.encoder, self.decoder, self.lin_softmax)

    @override
    def forward(self, X: Tensor, y: Tensor, is_train: bool = False) -> Tensor:
        _ = self.encoder(X, is_train)
        out = self.decoder(y, is_train)
        out = self.lin_softmax(out, is_train)
        return out

    @override
    def backward(self, d_out: Tensor) -> Tensor:
        d_out = self.lin_softmax.backward(d_out)
        _ = self.decoder.backward(d_out)
        d_out = self.encoder.backward(d_out)
        return d_out
