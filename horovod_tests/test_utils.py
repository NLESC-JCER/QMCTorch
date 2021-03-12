"""Module containing some utilities for testing."""

from pathlib import Path
import pkg_resources as pkg

__all__ = ["PATH_QMCTORCH", "PATH_TEST", "second_derivative"]

# Environment data
PATH_QMCTORCH = Path(pkg.resource_filename('qmctorch', ''))
ROOT = PATH_QMCTORCH.parent

PATH_TEST = ROOT / "horovod_tests"


def second_derivative(xm1, x0, xp1, eps):
    return (xm1 - 2 * x0 + xp1) / eps / eps
