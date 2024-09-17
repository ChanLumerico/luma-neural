"""
`neural.model`
--------------
Neural models are computational systems designed to learn patterns and make 
predictions based on data. They consist of multiple neural layers organized 
in a structured way, allowing the model to process information, extract 
features, and generate outputs. These models can handle a wide range of tasks, 
from image recognition to natural language processing, by learning from data 
and improving performance over time through training.

"""

from .simple import *
from .img_clf import *


MODLES: tuple = simple.__all__ + img_clf.__all__
NUM_MODELS: int = len(MODLES)
