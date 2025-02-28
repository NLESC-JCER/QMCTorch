import subprocess
import os
from ..__version__ import __version__


def get_git_tag() -> str:
    """
    Retrieves the current Git tag for the repository.

    This function determines the directory of the current file, then executes
    a Git command to describe the current commit with the most recent tag.

    Returns:
        str: The Git tag string representing the current state of the repository.
    """
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        gittag = (
            subprocess.check_output(["git", "describe", "--always"], cwd=cwd)
            .decode("utf-8")
            .strip("\n")
        )
        return __version__ + " - " + gittag
    except:
        return __version__ + " - hash commit not found"
