import os
import sys

from .file_utils import _BaseLazyModule, is_torch_available, is_tf_available

_import_structure = {
    "file_utils": ["is_torch_available", "is_tf_available"]
}

if is_torch_available():
    _import_structure["models"] = ["BertEmbeddings"]
else:
    _import_structure["utils.dummy_pt"] = ["BertEmbeddings"]

if is_tf_available():
    _import_structure["model_tf"] = ["TFBertEmbeddings"]
else:
    _import_structure["utils.dummy_tf"] = ["TFBertEmbeddings"]


class _LazyModule(_BaseLazyModule):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """
    __file__ = globals()["__file__"]
    __path__ = [os.path.dirname(__file__)]

    def _get_module(self, module_name: str):
        import importlib
        return importlib.import_module("." + module_name, self.__name__)

sys.modules[__name__] = _LazyModule(__name__, _import_structure)
