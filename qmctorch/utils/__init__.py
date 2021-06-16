"""Utils module API."""

from .algebra_utils import bdet2, bproj, btrace
from .hdf5_utils import (add_group_attr, dump_to_hdf5, load_from_hdf5,
                         register_extra_attributes, bytes2str)
from .interpolate import (InterpolateAtomicOrbitals,
                          InterpolateMolecularOrbitals)
from .plot_data import (plot_block, plot_blocking_energy,
                        plot_correlation_coefficient, plot_correlation_time,
                        plot_data, plot_energy,
                        plot_integrated_autocorrelation_time,
                        plot_walkers_traj)
from .stat_utils import (blocking, correlation_coefficient,
                         integrated_autocorrelation_time)
from .torch_utils import (DataSet, Loss, OrthoReg, fast_power,
                          set_torch_double_precision,
                          set_torch_single_precision,
                          diagonal_hessian, gradients)

__all__ = ['plot_energy', 'plot_data', 'plot_block',
           'plot_walkers_traj',
           'plot_correlation_time',
           'plot_autocorrelation',
           'set_torch_double_precision',
           'set_torch_single_precision',
           'DataSet', 'Loss', 'OrthoReg',
           'dump_to_hdf5', 'load_from_hdf5',
           'bytes2str',
           'register_extra_attributes',
           'fast_power',
           'InterpolateMolecularOrbitals',
           'InterpolateAtomicOrbitals',
           'btrace', 'bdet2', 'bproj',
           'diagonal_hessian', 'gradients']
