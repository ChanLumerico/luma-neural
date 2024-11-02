from typing import override

from luma.core.super import Supervised
from luma.interface.typing import Tensor
from luma.interface.util import InitUtil

from luma.neural.base import NeuralModel
from luma.neural import layer as nl
from luma.neural import block as nb

from ..types import SequenceToSequence


class _Transformer_Base(NeuralModel, SequenceToSequence, Supervised):
    def __init__(
        self,
        seq_length_arr: list[int],  # [L_src, L_tgt]
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        out_features: int = 30000,
        batch_size: int = 64,
        n_epochs: int = 10,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 3,
        shuffle: bool = True,
        random_state: int | None = None,
        deep_verbose: bool = False,
    ) -> None:
        self.seq_length_arr = seq_length_arr
        self.activation = activation
        self.initializer = initializer
        self.out_features = out_features
        self.lambda_ = lambda_
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
            6, 512, 2048, 8, self.enc_mask_, **base_args
        )
        self.decoder = nb.TransformerBlock.DecoderStack(
            6,
            512,
            2048,
            8,
            self.encoder[-1][1],
            self.dec_mask_self_,
            self.dec_mask_cross_,
            **base_args
        )
        self.lin_softmax = nl.Sequential(
            nl.DenseND(
                512,
                self.out_features,
                -1,
                self.initializer,
                None,
                self.lambda_,
                self.random_state,
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


class _Transformer_Big(NeuralModel, SequenceToSequence, Supervised): ...
