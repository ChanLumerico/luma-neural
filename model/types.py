from abc import ABCMeta


class ImageClassifier(metaclass=ABCMeta):
    """
    Abstract base class for image classification models.

    This class serves as an annotation for building custom
    image classification models.

    The `ImageClassifier` class is intended to be inherited by
    specific image classification models, which will implement
    the necessary methods for training, evaluation, and inference.
    """


class SequenceToSequence(metaclass=ABCMeta):
    """
    Abstract base class for sequence-to-sequence models.

    This class serves as an annotation for building custom
    sequence-to-sequence models, typically used in tasks such as
    machine translation, text summarization, and conversational
    response generation.

    The `SequenceToSequence` class is intended to be inherited by
    specific sequence-to-sequence models, which will implement
    the necessary methods for training, evaluation, and inference.
    """
