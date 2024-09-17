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

from types import ModuleType

from .simple import *
from .img_clf import *


SUB_MODULES: tuple[ModuleType] = (simple, img_clf)
MODELS: tuple = ()

for module in SUB_MODULES:
    if hasattr(module, "__all__"):
        MODELS += module.__all__

NUM_MODELS: int = len(MODELS)
