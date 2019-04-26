#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the pyCHAMP module.
"""
import autograd.numpy as np

from pyCHAMP.wavefunction.wf_base import WF
from pyCHAMP.optimizer.minimize import MINIMIZE
from pyCHAMP.sampler.metropolis import METROPOLIS
from pyCHAMP.solver.vmc import VMC

import unittest

class HarmOsc1D(WF):

    def __init__(self,nelec,ndim):
        WF.__init__(self, nelec, ndim)

    def values(self,parameters,pos):
        ''' Compute the value of the wave function.

        Args:
            parameters : parameters of th wf
            x: position of the electron

        Returns: values of psi
        '''
    
        beta = parameters[0]
        return np.exp(-beta*pos**2).reshape(-1,1)

    def nuclear_potential(self,pos):
        return 0.5*pos**2 

    def electronic_potential(self,pos):
        return 0


class TestHarmonicOscillator1D(unittest.TestCase):

    def setUp(self):
        self.wf = HarmOsc1D(nelec=1, ndim=1)
        self.sampler = METROPOLIS(nwalkers=1000, nstep=1000, step_size = 3, nelec=1, ndim=1, domain = {'min':-2,'max':2})
        self.optimizer = MINIMIZE(method='bfgs', maxiter=25, tol=1E-4)

    def test_vmc_single_point(self):
        vmc = VMC(wf=self.wf, sampler=self.sampler, optimizer=None)
        opt_param = [0.5]
        _, e, s = vmc.single_point(opt_param)
        assert np.allclose([e,s],[0.5,0],atol=1E-3)

    def test_vmc_opt(self):
        vmc = VMC(wf=self.wf, sampler=self.sampler, optimizer=self.optimizer)
        init_param = [1.25]
        vmc.optimize(init_param)
        vf =  vmc.history['variance'][-1]
        assert np.allclose([vf],[0.0],atol=1E-3)


if __name__ == "__main__":
    unittest.main()