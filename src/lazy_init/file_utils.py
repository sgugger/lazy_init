import os

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
        import torch

        _torch_available = True  # pylint: disable=invalid-name
    else:
        _torch_available = False
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        import tensorflow as tf

        assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
        _tf_available = True  # pylint: disable=invalid-name
    else:
        _tf_available = False
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


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