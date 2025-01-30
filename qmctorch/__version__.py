import subprocess
import os

__version__ = "0.3.2"

cwd = os.path.dirname(os.path.abspath(__file__))
git_describe_tag = subprocess.check_output(["git", "describe", "--tags"], cwd=cwd).decode("utf-8").strip("\n")