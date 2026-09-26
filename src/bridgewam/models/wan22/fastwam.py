"""Compatibility module; implementation: ``bridgewam.models.wan22.bridgewam``."""

import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("bridgewam.models.wan22.bridgewam")
