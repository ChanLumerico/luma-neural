"""
Sequence-to-Sequence Models
---------------------------

This module contains a collection of neural network architectures 
commonly used for sequence-to-sequence tasks. These models are 
designed to process input sequences and generate corresponding 
output sequences, making them suitable for applications such as 
machine translation, text summarization, and conversational response 
generation.

The models in this module vary in complexity, depth, and computational 
efficiency, providing options for a range of use cases, from 
resource-limited deployments to high-performance, large-scale systems.

The models included in this module cover diverse design paradigms, 
such as recurrent neural networks (RNNs), transformers, and attention 
mechanisms, ensuring adaptability for various sequence-based tasks.
"""

from . import transformer


__all__ = ("Transformer_Base", "Transformer_Big")


class Transformer_Base(transformer._Transformer_Base):
    """
    The Transformer-Base model uses an encoder-decoder structure
    with six layers in each to handle sequence-to-sequence tasks
    like translation. Each layer applies multi-head self-attention
    and feedforward networks, capturing relationships across
    sequences simultaneously.

    Specs
    -----
    Input/Output Shapes:
    ```
    Tensor[-1, -1, 512] -> Tensor[-1, -1, 37000]
    ```
    Parameter Size:
    ```
    62,984,192 weight, 104,584 biases -> 63,088,776 params
    ```
    Components
    ----------
    Blocks Used:
    ```
    TransformerBlock.EncoderStack(),
    TransformerBlock.DecoderStack()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=37000
        Number of output features
    `batch_size` : int, default=64
        Size of a single mini-batch
    `n_epochs` : int, default=10
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=3
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Vaswani, Ashish, et al. “Attention Is All You Need.”
    Advances in Neural Information Processing Systems,
    vol. 30, 2017.

    """


class Transformer_Big(transformer._Transformer_Big):
    """
    The Transformer-Big model is a larger variant of the original
    Transformer, featuring more layers and increased hidden dimensions.
    This expanded capacity allows it to capture more complex patterns
    in data, improving performance on sequence-to-sequence tasks like
    language translation while requiring significantly more
    computational resources.

    Specs
    -----
    Input/Output Shapes:
    ```
    Tensor[-1, -1, 1024] -> Tensor[-1, -1, 37000]
    ```
    Parameter Size:
    ```
    214,048,768 weights, 172,168 biases -> 214,220,936 params
    ```
    Components
    ----------
    Blocks Used:
    ```
    TransformerBlock.EncoderStack(),
    TransformerBlock.DecoderStack()
    ```
    Arguments
    ---------
    `activation` : callable, default=Activation.ReLU
        Type of activation function
    `initializer` : InitStr, default=None
        Type of weight initializer
    `out_features` : int, default=37000
        Number of output features
    `batch_size` : int, default=64
        Size of a single mini-batch
    `n_epochs` : int, default=10
        Number of epochs for training
    `valid_size` : float, default=0.1
        Fractional size of validation set
    `lambda_` : float, default=0.0
        L2 regularization strength
    `early_stopping` : bool, default=False
        Whether to early-stop the training when the valid score stagnates
    `patience` : int, default=3
        Number of epochs to wait until early-stopping
    `shuffle` : bool, default=True
        Whethter to shuffle the data at the beginning of every epoch

    References
    ----------
    [1] Vaswani, Ashish, et al. “Attention Is All You Need.”
    Advances in Neural Information Processing Systems,
    vol. 30, 2017.

    """
