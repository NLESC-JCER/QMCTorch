import subprocess
import os
cwd = os.path.dirname(os.path.abspath(__file__))
gittag = subprocess.check_output(["git", "describe", "--tags"], cwd=cwd).decode("utf-8").strip("\n")