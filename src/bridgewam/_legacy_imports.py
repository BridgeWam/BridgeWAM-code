"""Lazy, identity-preserving imports for the historical ``fastwam`` namespace.

Only the canonical modules execute. In particular, loading old pickle GLOBALs,
Hydra targets, or monkeypatch paths must not create a second set of model classes.
Keep this module independent of torch and optional dataset/simulator packages.
"""

from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder
import sys


_RENAMES = {
    "models.wan22.fastwam": "models.wan22.bridgewam",
    "models.wan22.fastwam_joint": "models.wan22.bridgewam_joint",
    "models.wan22.fastwam_idm": "models.wan22.bridgewam_idm",
    "models.wan22.aclation_bridgewam": "models.wan22.ablation_bridgewam",
    "datasets.lerobot.processors.fastwam_processor":
        "datasets.lerobot.processors.bridgewam_processor",
}


def _canonical_name(fullname):
    suffix = fullname.removeprefix("fastwam.")
    for old, new in _RENAMES.items():
        if suffix == old or suffix.startswith(old + "."):
            suffix = new + suffix[len(old):]
            break
    return "bridgewam." + suffix


class _AliasLoader(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Replace the placeholder, rather than returning the canonical module
        # from create_module: importlib must not overwrite its canonical spec.
        sys.modules[module.__name__] = import_module(_canonical_name(module.__name__))


class _FastWAMFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("fastwam."):
            return None
        canonical = util.find_spec(_canonical_name(fullname))
        if canonical is None:
            return None
        return util.spec_from_loader(
            fullname, _AliasLoader(),
            is_package=canonical.submodule_search_locations is not None,
        )


def install():
    if not any(isinstance(finder, _FastWAMFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FastWAMFinder())
