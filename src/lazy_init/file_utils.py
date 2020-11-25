import importlib.util
import os


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
