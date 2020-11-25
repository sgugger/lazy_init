from ..file_utils import requires_tf

class TFBertEmbeddings:
    def __init__(self, *args, **kwargs):
        requires_tf(self)