"""Legacy imports for BridgeWAM; all implementation lives in ``bridgewam``."""

from bridgewam._legacy_imports import install as _install

_install()
del _install


def __getattr__(name):
    if name in {"BridgeWAM", "FastWAM"}:
        from bridgewam.models.wan22.bridgewam import BridgeWAM
        return BridgeWAM
    raise AttributeError(name)
