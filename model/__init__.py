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
from typing import Any, Never

from .simple import *
from .img_clf import *


SUB_MODULES: tuple[ModuleType] = (simple, img_clf)
MODELS: tuple[str] = ()

for module in SUB_MODULES:
    if hasattr(module, "__all__"):
        MODELS += module.__all__

NUM_MODELS: int = len(MODELS)


def debug_models(submodules: list[str] | None = None) -> None:
    all_models = []
    if submodules is None:
        all_models.extend(MODELS)
    else:
        for module_name in submodules:
            if module_name not in globals().keys():
                raise AttributeError(f"'{module_name}' is an invalid submodule!")

            submodule = globals()[module_name]
            all_models.extend(submodule.__all__)

    all_models = [m_name for m_name in all_models if get_model(m_name).do_debug]

    print(f"Start debugging for {len(all_models)} model(s)...")
    print("=" * 75)

    fail_count = 0
    failed_models: list[str] = []
    for i, model_name in enumerate(all_models, start=1):
        model: type = get_model(model_name)
        print(f"[{i}/{len(all_models)}]", end=" ")

        try:
            tmp = model()
            del tmp
        except Exception:
            fail_count += 1
            failed_models.append(model.__name__)
            print(f"'{model_name}' instantiation failed!", end=", ")
        else:
            print(f"'{model_name}' instantiation succeeded", end=", ")

        print(f"fail count: {fail_count}")

    print("=" * 75)
    print(f"Success: {len(all_models) - fail_count}, Failure: {fail_count}")
    print(f"Failed Models: {failed_models}")


def _get_alt_name(name: str) -> str:
    return name.lower().replace("_", "-")


def get_model(name: str) -> type | None:
    for model_name in MODELS:
        alt_name = _get_alt_name(model_name)
        if name == model_name or name == alt_name:
            return globals()[model_name]


def get_model_instance(name: str, **kwargs) -> object:
    model = get_model(name)
    if model is None:
        raise ValueError(f"'{name}' is an invalid or unsupported model!")
    return model(**kwargs)


import json
import os

reg_path: str = r"luma/neural/model/registry.json"


def register_model(name: str, **kwargs: Any) -> None:
    if load_model_registry(name) is not None:
        print(
            f"Model '{name}' is already registered.",
            "Skipping registration.",
        )
        return

    with open(reg_path, "r") as f:
        try:
            model_data = json.load(f)
        except json.JSONDecodeError:
            model_data = []

    model = get_model(name)
    if not model.do_register:
        raise ValueError(
            f"Model '{model.__name__}' is not supposed to be registered."
            + " Skipping registration.",
        )

    model_inst = model()
    new_entry = {
        "name": model.__name__,
        "weights": model_inst.param_size[0],
        "biases": model_inst.param_size[1],
        "params": sum(model_inst.param_size),
        "layers": model_inst.layer_count,
        **kwargs,
    }

    model_data.append(new_entry)
    with open(reg_path, "w") as f:
        json.dump(model_data, f, indent=4)


def _check_reg_path() -> Never:
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"[Fatal] Model registry file not found!")


def load_model_registry(name: str) -> dict | None:
    _check_reg_path()
    with open(reg_path, "r") as f:
        try:
            model_data = json.load(f)
        except json.JSONDecodeError:
            model_data = []

    for entry in model_data:
        entry_name = entry["name"]
        if entry_name == name or _get_alt_name(entry_name) == name:
            return entry


def load_entire_registry() -> list[dict]:
    _check_reg_path()
    with open(reg_path, "r") as f:
        model_data = json.load(f)

    return model_data
