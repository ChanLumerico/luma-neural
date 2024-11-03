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
    TODO: Docstring for `Transformer-Base`
    """


class Transformer_Big(transformer._Transformer_Big):
    """
    TODO: Docstring for `Transformer-Big`
    """
