# -*- coding: utf-8 -*-
from .classdef import C2

def _added_to_val_w(instance: C2, to_add: float) -> C2:
    return C2(instance.val + to_add)