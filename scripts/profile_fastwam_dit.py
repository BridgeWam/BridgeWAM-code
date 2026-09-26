"""Legacy launcher for profile_bridgewam_dit.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("profile_bridgewam_dit.py")), run_name="__main__")
else:
    from scripts.profile_bridgewam_dit import *
