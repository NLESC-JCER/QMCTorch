import torch
import numpy as np


class OrbitalConfigurations(object):

    def __init__(self, configs, mol):
        self.configs = configs.lower()
        self.mol = mol

    def get_configs(self):
        """Get the configuratio in the CI expansion

        Args:
            configs (str): name of the configs we want
            mol (mol object): molecule object

        Returns:
            tuple(torch.LongTensor,torch.LongTensor): the spin up/spin down
            electronic confs
        """
        if isinstance(self.configs, torch.Tensor):
            return self.configs

        elif self.configs == 'ground_state':
            return self._get_ground_state_config()

        elif self.configs.startswith('cas'):
            nelec, norb = eval(self.configs.lstrip("cas"))
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_cas_config(self.mol, nocc, nvirt)

        elif self.configs.startswith('singlet'):
            nelec, norb = eval(self.configs.lstrip("singlet"))
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_singlet_state_config(self.mol, nocc, nvirt)

        else:
            print(configs, " not recognized as valid configuration")
            print('Options are : ground_state or singlet(nocc,nvirt)')
            raise ValueError()

    def _get_ground_state_config(self):
        """Return only the ground state configuration

        Args:
            mol (mol): mol object

        Returns:
            tuple(torch.LongTensor,torch.LongTensor): the spin up/spin down
            electronic confs
        """
        conf = (torch.LongTensor([np.array(range(self.mol.nup))]),
                torch.LongTensor([np.array(range(self.mol.ndown))]))
        return conf

    def _get_singlet_state_config(self, nocc, nvirt):
        """Get the confs of the singlet conformations

        Args:
            mol (mol): mol object
            nocc (int): number of occupied orbitals in the active space
            nvirt (int): number of virtual orbitals in the active space
        """

        _gsup, _gs_down = list(range(self.mol.nup)), list(
            range(self.mol.ndown))
        cup, cdown = [_gsup], [_gs_down]

        for ivirt in range(self.mol.nup, self.mol.nup+nvirt, 1):
            for iocc in range(self.mol.nup-1, self.mol.nup-1-nocc, -1):

                _xt = self._create_excitation(_gs.copy(), iocc, ivirt)
                cup, cdown = self._append_excitations(cup, cdown, _xt, _gs)
                cup, cdown = self._append_excitations(cup, cdown, _gs, _xt)

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_cas_config(self, nocc, nvirt):
        """Get the confs of the CAS

        Args:
            mol (mol): molecule object
            nelec (int): number of electrons in the active space
            norb (int) : number of orbitals in the active space
        """

        cup, cdown = [], []

        for ivirt_up in range(self.mol.nup-1, self.mol.nup+nvirt, 1):
            for iocc_up in range(self.mol.nup-1, self.mol.nup-1-nocc, -1):

                for ivirt_down in range(self.mol.ndown-1, self.mol.ndown+nvirt, 1):
                    for iocc_up in range(self.mol.ndown-1, self.mol.ndown-1-nocc, -1):

                        _xt_up = self._create_excitation(_gs.copy(), iocc_up, ivirt_up)
                        _xt_down = self._create_excitation(
                            _gs.copy(), iocc_down, ivirt_down)
                        cup, cdown = self._append_excitations(
                            cup, cdown, _xt_up, _xt_down)

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_orb_number(self, nelec, norb):

        _gsup, _gs_down = list(range(self.mol.nup)), list(
            range(self.mol.ndown))

        nocc = nelec // 2
        nvirt = norb // 2 - nocc
        return nocc, nvirt

    @staticmethod
    def _create_excitation(conf, iocc, ivirt):
        """promote an electron from iocc to ivirt

        Args:
            conf (list): index of the occupied orbitals
            iocc (int): index of the occupied orbital
            ivirt (int): index of the virtual orbital

        Returns:
            list: new configuration
        """
        conf.pop(iocc)
        conf += [ivirt]
        return conf

    @staticmethod
    def _append_excitations(cup, cdown, new_cup, new_cdown):
        """Append new excitations

        Args:
            cup (list): configurations of spin up
            cdown (list): configurations of spin down
            new_cup (list): new spin up confs
            new_cdown (list): new spin down confs
        """

        cup.append(new_cup)
        cdown.append(new_cdown)
        return cup, cdown
