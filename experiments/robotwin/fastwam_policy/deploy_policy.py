"""Legacy RoboTwin policy entrypoint."""
import sys
from pathlib import Path

# RoboTwin imports this through policy/fastwam_policy, outside the repo root.
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from experiments.robotwin.bridgewam_policy.deploy_policy import *
