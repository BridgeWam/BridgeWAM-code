"""BridgeWAM training and inference package."""

__all__ = ["BridgeWAM", "BridgeWAMJoint", "BridgeWAMIDM"]


def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        suffix = {"BridgeWAM": "", "BridgeWAMJoint": "_joint", "BridgeWAMIDM": "_idm"}[name]
        return getattr(import_module("bridgewam.models.wan22.bridgewam" + suffix), name)
    raise AttributeError(name)
