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


class ObjectDetector(metaclass=ABCMeta):
    """
    Abstract base class for object detection models.

    This class serves as an annotation for building custom
    object detection models.

    The `ObjectDetector` class is intended to be inherited by
    specific object detection models, which will implement
    the necessary methods for object detection tasks such as
    bounding box prediction, class identification, and confidence
    scoring.
    """