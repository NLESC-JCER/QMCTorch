"""Utils module API."""

from .algebra_utils import bdet2, bproj, btrace
from .provenance import get_git_tag
from .hdf5_utils import (
    add_group_attr,
    dump_to_hdf5,
    load_from_hdf5,
    register_extra_attributes,
    bytes2str,
)
from .interpolate import InterpolateAtomicOrbitals, InterpolateMolecularOrbitals


from .stat_utils import (
    blocking,
    correlation_coefficient,
    integrated_autocorrelation_time,
)

from .torch_utils import (
    DataSet,
    DataLoader,
    fast_power,
    set_torch_double_precision,
    set_torch_single_precision,
    diagonal_hessian,
    gradients,
    hessian,
)

__all__ = [
    "set_torch_double_precision",
    "set_torch_single_precision",
    "DataSet",
    "DataLoader",
    "add_group_attr",
    "dump_to_hdf5",
    "load_from_hdf5",
    "bytes2str",
    "register_extra_attributes",
    "fast_power",
    "InterpolateMolecularOrbitals",
    "InterpolateAtomicOrbitals",
    "btrace",
    "bdet2",
    "bproj",
    "diagonal_hessian",
    "gradients",
    "hessian",
    "blocking",
    "correlation_coefficient",
    "integrated_autocorrelation_time",
    "get_git_tag",
]
