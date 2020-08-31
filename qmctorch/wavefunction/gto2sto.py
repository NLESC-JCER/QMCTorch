from . import Orbital
from .. import log
import numpy as np
from copy import deepcopy
from scipy.optimize import curve_fit
import torch


def gto2sto(wf, plot=False):
    """Fits the AO GTO to AO STO.
        The SZ sto tht have only one basis function per ao
    """

    assert(wf.ao.radial_type.startswith('gto'))
    assert(wf.ao.harmonics_type == 'cart')

    log.info('  Fit GTOs to STOs  : ')

    def sto(x, norm, alpha):
        """Fitting function."""
        return norm * np.exp(-alpha * np.abs(x))

    # shortcut for nao
    nao = wf.mol.basis.nao

    # create a new mol and a new basis
    new_mol = deepcopy(wf.mol)
    basis = deepcopy(wf.mol.basis)

    # change basis to sto
    basis.radial_type = 'sto_pure'
    basis.nshells = wf.ao.nao_per_atom.numpy()

    # reset basis data
    basis.index_ctr = np.arange(nao)
    basis.bas_coeffs = np.ones(nao)
    basis.bas_exp = np.zeros(nao)
    basis.bas_norm = np.zeros(nao)
    basis.bas_kr = np.zeros(nao)
    basis.bas_kx = np.zeros(nao)
    basis.bas_ky = np.zeros(nao)
    basis.bas_kz = np.zeros(nao)

    # 2D fit space
    x = torch.linspace(-5, 5, 501)

    # compute the values of the current AOs using GTO BAS
    pos = x.reshape(-1, 1).repeat(1, wf.ao.nbas).to(wf.device)
    gto = wf.ao.norm_cst * torch.exp(-wf.ao.bas_exp*pos**2)
    gto = gto.unsqueeze(1).repeat(1, wf.nelec, 1)
    ao = wf.ao._contract(gto)[
        :, 0, :].detach().cpu().numpy()

    # loop over AOs
    for iorb in range(wf.ao.norb):

        # fit AO with STO
        xdata = x.numpy()
        ydata = ao[:, iorb]
        popt, pcov = curve_fit(sto, xdata, ydata)

        # store new exp/norm
        basis.bas_norm[iorb] = popt[0]
        basis.bas_exp[iorb] = popt[1]

        # determine k values
        basis.bas_kx[iorb] = wf.ao.harmonics.bas_kx[wf.ao.index_ctr == iorb].unique(
        ).item()
        basis.bas_ky[iorb] = wf.ao.harmonics.bas_ky[wf.ao.index_ctr == iorb].unique(
        ).item()
        basis.bas_kz[iorb] = wf.ao.harmonics.bas_kz[wf.ao.index_ctr == iorb].unique(
        ).item()

        # plot if necessary
        if plot:
            plt.plot(xdata, ydata)
            plt.plot(xdata, sto(xdata, *popt))
            plt.show()

    # update basis in new mole
    new_mol.basis = basis

    # returns new orbital instance
    return Orbital(new_mol, configs=wf.configs_method,
                   kinetic=wf.kinetic_method,
                   use_jastrow=wf.use_jastrow,
                   jastrow_type=wf.jastrow_type,
                   cuda=wf.cuda,
                   include_all_mo=wf.include_all_mo)
