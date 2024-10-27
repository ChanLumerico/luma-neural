from typing import ClassVar, Literal, Self, override

from luma.core.super import Estimator, Evaluator
from luma.interface.typing import Matrix, Tensor, Vector
from luma.interface.util import InitUtil

from luma.neural.base import NeuralModel
from luma.neural import layer as nl
from luma.neural import block as nb


__all__ = ("SimpleMLP", "SimpleCNN", "SimpleTransformer")


class SimpleMLP(Estimator, NeuralModel):
    """
    An MLP (Multilayer Perceptron) is a type of artificial neural network
    composed of at least three layers: an input layer, one or more hidden
    layers, and an output layer. Each layer consists of nodes, or neurons,
    which are fully connected to the neurons in the next layer. MLPs use a
    technique called backpropagation for learning, where the output error
    is propagated backwards through the network to update the weights.
    They are capable of modeling complex nonlinear relationships between
    inputs and outputs. MLPs are commonly used for tasks like classification,
    regression, and pattern recognition.

    Structure
    ---------
    ```py
    (nl.Dense -> Activation -> nl.Dropout) -> ... -> nl.Dense
    ```
    Parameters
    ----------
    `in_features` : int
        Number of input features
    `out_features` : int
        Number of output features
    `hidden_layers` : int of list of int
        Numbers of the features in hidden layers (int for a single layer)
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `initializer` : InitStr, default=None
        Type of weight initializer
    `activation` : callable
        Type of activation function
    `dropout_rate` : float, default=0.5
        nl.Dropout rate
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    Notes
    -----
    - If the data or the target is a 1D-Array(`Vector`), reshape it into a
        higher dimensional array.

    - For classification tasks, the target vector `y` must be
        one-hot encoded.

    """

    do_debug: ClassVar[bool] = False
    do_register: ClassVar[bool] = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: list[int] | int,
        *,
        activation: callable,
        initializer: InitUtil.InitStr = None,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        dropout_rate: float = 0.5,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.initializer = initializer
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.lambda_ = lambda_
        self.shuffle = shuffle
        self.random_state = random_state
        self.fitted_ = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = nl.Sequential()

        if isinstance(self.hidden_layers, int):
            self.hidden_layers = [self.hidden_layers]

        self.feature_sizes_ = [
            self.in_features,
            *self.hidden_layers,
            self.out_features,
        ]
        self.feature_shapes_ = self._get_feature_shapes(self.feature_sizes_)

        self.set_param_ranges(
            {
                "in_features": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        for i, (in_, out_) in enumerate(self.feature_shapes_):
            self.model += nl.Dense(
                in_,
                out_,
                initializer=self.initializer,
                lambda_=self.lambda_,
                random_state=self.random_state,
            )
            if i < len(self.feature_shapes_) - 1:
                self.model += self.activation()
                self.model += nl.Dropout(
                    dropout_rate=self.dropout_rate,
                    random_state=self.random_state,
                )

    def fit(self, X: Matrix, y: Matrix) -> Self:
        return super(SimpleMLP, self).fit_nn(X, y)

    @override
    def predict(self, X: Matrix, argmax: bool = True) -> Matrix | Vector:
        return super(SimpleMLP, self).predict_nn(X, argmax)

    @override
    def score(
        self, X: Matrix, y: Matrix, metric: Evaluator, argmax: bool = True
    ) -> float:
        return super(SimpleMLP, self).score_nn(X, y, metric, argmax)


class SimpleCNN(Estimator, NeuralModel):
    """
    A Convolutional Neural Network (CNN) is a type of deep neural network
    primarily used in image recognition and processing that is particularly
    powerful at capturing spatial hierarchies in data. A CNN automatically
    detects important features without any human supervision using layers
    with convolving filters that pass over the input image and compute outputs.
    These networks typically include layers such as convolutional layers,
    pooling layers, and fully connected layers that help in reducing the
    dimensions while retaining important features.

    Structure
    ---------
    ```py
    nb.ConvBlock2D -> ... -> nl.Flatten -> nb.DenseBlock -> ... -> nl.Dense
    ```
    Parameters
    ----------
    `in_channels_list` : int or list of int
        List of input channels for convolutional blocks
    `in_features_list` : int or list of int
        List of input features for dense blocks
    `out_channels` : int
        Output channels for the last convolutional layer
    `out_features` : int
        Output features for the last dense layer
    `filter_size` : int
        Size of filters for convolution layers
    `activation` : callable
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer (None for dense layers)
    `padding` : {"same", "valid"}, default="same"
        Padding strategy
    `stride` : int, default=1
        Step size of filters during convolution
    `do_batch_norm` : bool, default=True
        Whether to perform batch normalization
    `momentum` : float, default=0.9
        Momentum for batch normalization
    `do_pooling` : bool, default=True
        Whether to perform pooling
    `pool_filter_size` : int, default=2
        Size of filters for pooling layers
    `pool_stride` : int, default=2
        Step size of filters during pooling
    `pool_mode` : {"max", "avg"}, default="max"
        Pooling strategy (default `max`)
    `do_dropout` : bool, default=True
        Whether to perform dropout
    `dropout_rate` : float, default=0.5
        nl.Dropout rate
    `batch_size` : int, default=100
        Size of a single mini-batch
    `n_epochs` : int, default=100
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=10
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    Notes
    -----
    - Input `X` must have the shape of 4D-array(`Tensor`)

    - For classification tasks, the target vector `y` must be
        one-hot encoded.

    """

    do_debug: ClassVar[bool] = False
    do_register: ClassVar[bool] = False

    def __init__(
        self,
        in_channels_list: list[int] | int,
        in_features_list: list[int] | int,
        out_channels: int,
        out_features: int,
        filter_size: int,
        *,
        activation: callable,
        initializer: InitUtil.InitStr = None,
        padding: Literal["same", "valid"] = "same",
        stride: int = 1,
        do_batch_norm: bool = True,
        momentum: float = 0.9,
        do_pooling: bool = True,
        pool_filter_size: int = 2,
        pool_stride: int = 2,
        pool_mode: Literal["max", "avg"] = "max",
        do_dropout: bool = True,
        dropout_rate: float = 0.5,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        self.in_channels_list = in_channels_list
        self.in_features_list = in_features_list
        self.out_channels = out_channels
        self.out_features = out_features
        self.filter_size = filter_size
        self.activation = activation
        self.initializer = initializer
        self.padding = padding
        self.stride = stride
        self.do_batch_norm = do_batch_norm
        self.momentum = momentum
        self.do_pooling = do_pooling
        self.pool_filter_size = pool_filter_size
        self.pool_stride = pool_stride
        self.pool_mode = pool_mode
        self.do_dropout = do_dropout
        self.dropout_rate = dropout_rate
        self.lambda_ = lambda_
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = nl.Sequential()

        if isinstance(self.in_channels_list, int):
            self.in_channels_list = [self.in_channels_list]
        if isinstance(self.in_features_list, int):
            self.in_features_list = [self.in_features_list]

        self.feature_sizes_ = [
            [*self.in_channels_list, self.out_channels],
            [*self.in_features_list, self.out_features],
        ]
        self.feature_shapes_ = [
            [*self._get_feature_shapes(self.feature_sizes_[0])],
            [*self._get_feature_shapes(self.feature_sizes_[1])],
        ]

        self.set_param_ranges(
            {
                "out_channels": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "filter_size": ("0<,+inf", int),
                "stride": ("0<,+inf", int),
                "momentum": ("0,1", None),
                "pool_filter_size": ("0<,+inf", int),
                "pool_stride": ("0<,+inf", int),
                "dropout_rate": ("0,1", None),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        for in_, out_ in self.feature_shapes_[0]:
            self.model += nb.ConvBlock2D(
                in_,
                out_,
                self.filter_size,
                activation=self.activation,
                initializer=self.initializer,
                padding=self.padding,
                stride=self.stride,
                lambda_=self.lambda_,
                do_batch_norm=self.do_batch_norm,
                momentum=self.momentum,
                do_pooling=self.do_pooling,
                pool_filter_size=self.pool_filter_size,
                pool_stride=self.pool_stride,
                pool_mode=self.pool_mode,
                random_state=self.random_state,
            )

        self.model += nl.Flatten()
        for i, (in_, out_) in enumerate(self.feature_shapes_[1]):
            if i < len(self.feature_shapes_[1]) - 1:
                self.model += nb.DenseBlock(
                    in_,
                    out_,
                    activation=self.activation,
                    lambda_=self.lambda_,
                    do_dropout=self.do_dropout,
                    dropout_rate=self.dropout_rate,
                    random_state=self.random_state,
                )
            else:
                self.model += nl.Dense(
                    in_,
                    out_,
                    lambda_=self.lambda_,
                    random_state=self.random_state,
                )

    @Tensor.force_dim(4)
    def fit(self, X: Tensor, y: Matrix) -> Self:
        return super(SimpleCNN, self).fit_nn(X, y)

    @override
    @Tensor.force_dim(4)
    def predict(self, X: Tensor, argmax: bool = True) -> Matrix | Vector:
        return super(SimpleCNN, self).predict_nn(X, argmax)

    @override
    @Tensor.force_dim(4)
    def score(
        self, X: Tensor, y: Matrix, metric: Evaluator, argmax: bool = True
    ) -> float:
        return super(SimpleCNN, self).score_nn(X, y, metric, argmax)


class SimpleTransformer(Estimator, NeuralModel):
    def __init__(
        self,
        n_encoders: int,
        n_decoders: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        out_features: int,
        encoder_mask: Tensor | None = None,
        decoder_mask_self: Tensor | None = None,
        decoder_mask_cross: Tensor | None = None,
        pos_max_length: int = 500,
        activation: callable = nl.Activation.ReLU,
        initializer: InitUtil.InitStr = None,
        batch_size: int = 100,
        n_epochs: int = 100,
        valid_size: float = 0.1,
        lambda_: float = 0.0,
        dropout_rate: float = 0.1,
        early_stopping: bool = False,
        patience: int = 10,
        shuffle: bool = True,
        random_state: int = None,
        deep_verbose: bool = False,
    ) -> None:
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.out_features = out_features
        self.encoder_mask = encoder_mask
        self.decoder_mask_self = decoder_mask_self
        self.decoder_mask_cross = decoder_mask_cross
        self.pos_max_length = pos_max_length
        self.activation = activation
        self.initializer = initializer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.valid_size = valid_size
        self.lambda_ = lambda_
        self.dropout_rate = dropout_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.shuffle = shuffle
        self.random_state = random_state
        self._fitted = False

        super().__init__(
            batch_size,
            n_epochs,
            valid_size,
            early_stopping,
            patience,
            shuffle,
            random_state,
            deep_verbose,
        )
        super().init_model()
        self.model = nl.Sequential()

        self.set_param_ranges(
            {
                "n_encoders": ("0<,+inf", int),
                "n_decoders": ("0<,+inf", int),
                "d_model": ("0<,+inf", int),
                "d_ff": ("0<,+inf", int),
                "n_heads": ("0<,+inf", int),
                "out_features": ("0<,+inf", int),
                "batch_size": ("0<,+inf", int),
                "n_epochs": ("0<,+inf", int),
                "valid_size": ("0<,<1", None),
                "dropout_rate": ("0,1", None),
                "lambda_": ("0,+inf", None),
                "patience": (f"0<,+inf", int),
            }
        )
        self.check_param_ranges()
        self.build_model()

    def build_model(self) -> None:
        base_args = dict(
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            activation=self.activation,
            initializer=self.initializer,
            dropout_rate=self.dropout_rate,
            lambda_=self.lambda_,
            random_state=self.random_state,
        )

        self.model += nb.TransformerBlock.EncoderStack(
            n_encoders=self.n_encoders,
            mask=self.encoder_mask,
            pos_max_length=self.pos_max_length,
            **base_args,
        )
        self.model += nb.TransformerBlock.DecoderStack(
            n_decoders=self.n_decoders,
            mask_self=self.decoder_mask_self,
            mask_enc_dec=self.decoder_mask_cross,
            **base_args,
        )

        # TODO: Further implementation begins from here
