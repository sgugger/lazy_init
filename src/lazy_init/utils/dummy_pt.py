from ..file_utils import requires_pytorch

class BertEmbeddings:
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
