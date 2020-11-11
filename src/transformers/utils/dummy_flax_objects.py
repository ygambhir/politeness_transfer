# This file is autogenerated by the command `make fix-copies`, do not edit.
from ..file_utils import requires_flax


class FlaxBertModel:
    def __init__(self, *args, **kwargs):
        requires_flax(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_flax(self)


class FlaxRobertaModel:
    def __init__(self, *args, **kwargs):
        requires_flax(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_flax(self)
