"""Compatibility module; implementation: ``bridgewam.datasets.lerobot.processors.bridgewam_processor``."""

import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("bridgewam.datasets.lerobot.processors.bridgewam_processor")
