"""Module containing some utilities for testing."""

from pathlib import Path
import pkg_resources as pkg

__all__ = ["PATH_QMCTORCH", "PATH_TEST"]

# Environment data
PATH_QMCTORCH = Path(pkg.resource_filename('qmctorch', ''))
ROOT = PATH_QMCTORCH.parent

PATH_TEST = ROOT / "tests"
