from ..file_utils import is_torch_available, is_tf_available

if is_torch_available():
    from .model_pt import BertEmbeddings

if is_tf_available():
    from .model_tf import TFBertEmbeddings
