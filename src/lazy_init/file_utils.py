import importlib.util
import os
import sys
from types import ModuleType
from typing import Any


def is_torch_available():
    torch_spec = importlib.util.find_spec("torch")
    return torch_spec is not None


def is_tf_available():
    # Does not make a version check since it does not actually try to import it.
    tf_spec = importlib.util.find_spec("tensorflow")
    return tf_spec is not None


PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""


TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
"""


def requires_pytorch(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_torch_available():
        raise ImportError(PYTORCH_IMPORT_ERROR.format(name))


def requires_tf(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_tf_available():
        raise ImportError(TENSORFLOW_IMPORT_ERROR.format(name))


class _BaseLazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    def __init__(self, name, import_structure):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(import_structure.values(), [])
    
    def __dir__(self):
        return super().__dir__() + self.__all__
    
    def __getattr__(self, name: str) -> Any:
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        raise NotImplementedError
