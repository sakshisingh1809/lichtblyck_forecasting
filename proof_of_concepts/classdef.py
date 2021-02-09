from __future__ import annotations
from .methoddef import _added_to_val


class C2:
    def __init__(self, val):
        self.val = val


C2.added_to_val = _added_to_val
