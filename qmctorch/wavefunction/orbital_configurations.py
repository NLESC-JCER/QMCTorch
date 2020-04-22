import torch


class OrbitalConfigurations(object):

    def __init__(self, mol):
        # self.mol = mol
        self.nup = mol.nup
        self.ndown = mol.ndown

    def get_configs(self, configs):
        """Get the configuratio in the CI expansion

        Args:
            configs (str): name of the configs we want
            mol (mol object): molecule object

        Returns:
            tuple(torch.LongTensor,torch.LongTensor): the spin up/spin down
            electronic confs
        """

        if isinstance(configs, str):
            configs = configs.lower()

        if isinstance(configs, torch.Tensor):
            return configs

        elif configs == 'ground_state':
            return self._get_ground_state_config()

        elif configs.startswith('cas('):
            nelec, norb = eval(configs.lstrip("cas"))
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_cas_config(nocc, nvirt, nelec)

        elif configs.startswith('single('):
            nelec, norb = eval(configs.lstrip("single"))
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_single_config(nocc, nvirt)

        elif configs.startswith('single_double('):
            nelec, norb = eval(configs.lstrip("single_double"))
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_single_double_config(nocc, nvirt)

        else:
            print(configs, " not recognized as valid configuration")
            print('Options are : ground_state')
            print('              single(nelec,norb)')
            print('              single_double(nelec,norb)')
            print('              cas(nelec,norb)')
            raise ValueError("Config error")

    def _get_ground_state_config(self):
        """Return only the ground state configuration

        Args:
            mol (mol): mol object

        Returns:
            tuple(torch.LongTensor,torch.LongTensor): the spin up/spin down
            electronic confs
        """
        _gs_up = list(range(self.nup))
        _gs_down = list(range(self.ndown))
        cup, cdown = [_gs_up], [_gs_down]
        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_single_config(self, nocc, nvirt):
        """Get the confs of the singlet conformations

        Args:
            mol (mol): mol object
            nocc (int): number of occupied orbitals in the active space
            nvirt (int): number of virtual orbitals in the active space
        """

        _gs_up = list(range(self.nup))
        _gs_down = list(range(self.ndown))
        cup, cdown = [_gs_up], [_gs_down]

        for iocc in range(
                self.nup - 1, self.nup - 1 - nocc, -1):
            for ivirt in range(self.nup, self.nup + nvirt, 1):
                _xt = self._create_excitation(
                    _gs_up.copy(), iocc, ivirt)
                cup, cdown = self._append_excitations(
                    cup, cdown, _xt, _gs_down)

                _xt = self._create_excitation(
                    _gs_down.copy(), iocc, ivirt)
                cup, cdown = self._append_excitations(
                    cup, cdown, _gs_up, _xt)

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_single_double_config(self, nocc, nvirt):
        """Get the confs of the singlet + double

        Args:
            nelec (int): number of electrons in the active space
            norb (int) : number of orbitals in the active space
        """

        _gs_up = list(range(self.nup))
        _gs_down = list(range(self.ndown))
        cup, cdown = self._get_single_config(nocc, nvirt)
        cup = cup.tolist()
        cdown = cdown.tolist()

        for iocc_up in range(
                self.nup - 1, self.nup - 1 - nocc, -1):
            for ivirt_up in range(
                    self.nup, self.nup + nvirt, 1):

                for iocc_down in range(
                        self.ndown - 1, self.ndown - 1 - nocc, -1):
                    for ivirt_down in range(
                            self.ndown, self.ndown + nvirt, 1):

                        _xt_up = self._create_excitation(
                            _gs_up.copy(), iocc_up, ivirt_up)
                        _xt_down = self._create_excitation(
                            _gs_down.copy(), iocc_down, ivirt_down)
                        cup, cdown = self._append_excitations(
                            cup, cdown, _xt_up, _xt_down)

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_cas_config(self, nocc, nvirt, nelec):
        """get confs of the CAS

        Args:
            nocc (int): number of occupied orbitals in the CAS
            nvirt ([type]): number of virt orbitals in the CAS
        """
        from itertools import combinations, product

        idx_low, idx_high = self.nup - nocc, self.nup + nvirt
        orb_index_up = range(idx_low, idx_high)
        idx_frz = list(range(idx_low))
        _cup = [idx_frz + list(l)
                for l in list(combinations(orb_index_up, nelec // 2))]

        idx_low, idx_high = self.nup - nocc - 1, self.nup + nvirt - 1

        _cdown = [
            idx_frz +
            list(l) for l in list(
                combinations(orb_index_up, nelec // 2))]

        confs = list(product(_cup, _cdown))
        cup, cdown = [], []

        for c in confs:
            cup.append(c[0])
            cdown.append(c[1])

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_orb_number(self, nelec, norb):
        """compute the number of occupied and virtual orbital
        __ PER SPIN __
        __ ONLY VALID For spin up/down ___
        Args:
            nelec (int): total number of elec in the CAS
            norb (int): total number of orb in the CAS

        Returns:
            [int,int]: number of occpuied/virtual orb per spi
        """

        nocc = nelec // 2
        nvirt = norb - nocc
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
