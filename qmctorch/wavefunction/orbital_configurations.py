import torch


class OrbitalConfigurations(object):

    def __init__(self, mol):
        # self.mol = mol
        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = self.nup + self.ndown
        self.norb = mol.basis.nmo

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
            self.sanity_check(nelec, norb)
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_cas_config(nocc, nvirt, nelec)

        elif configs.startswith('single('):
            nelec, norb = eval(configs.lstrip("single"))
            self.sanity_check(nelec, norb)
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_single_config(nocc, nvirt)

        elif configs.startswith('single_double('):
            nelec, norb = eval(configs.lstrip("single_double"))
            self.sanity_check(nelec, norb)
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_single_double_config(nocc, nvirt)

        else:
            print(configs, " not recognized as valid configuration")
            print('Options are : ground_state')
            print('              single(nelec,norb)')
            print('              single_double(nelec,norb)')
            print('              cas(nelec,norb)')
            raise ValueError("Config error")

    def sanity_check(self, nelec, norb):
        """Check if the number of elec/orb is consistent with the
           properties of the molecule

        Args:
            nelec (int): required number of electrons in config
            norb (int): required number of orb in config

        """
        if nelec > self.nelec:
            raise ValueError(
                'required number of electron in config too large')

        if norb > self.norb:
            raise ValueError(
                'required number of orbitals in config too large')

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
        """Get the confs of the single + double

        Args:
            nelec (int): number of electrons in the active space
            norb (int) : number of orbitals in the active space
        """

        _gs_up = list(range(self.nup))
        _gs_down = list(range(self.ndown))
        cup, cdown = self._get_single_config(nocc, nvirt)
        cup = cup.tolist()
        cdown = cdown.tolist()

        idx_occ_up = list(
            range(self.nup - 1, self.nup - 1 - nocc, -1))
        idx_vrt_up = list(range(self.nup, self.nup + nvirt, 1))

        idx_occ_down = list(range(
            self.ndown - 1, self.ndown - 1 - nocc, -1))
        idx_vrt_down = list(range(self.ndown, self.ndown + nvirt, 1))

        # ground, single and double with 1 elec excited per spin
        for iocc_up in idx_occ_up:
            for ivirt_up in idx_vrt_up:

                for iocc_down in idx_occ_down:
                    for ivirt_down in idx_vrt_down:

                        _xt_up = self._create_excitation(
                            _gs_up.copy(), iocc_up, ivirt_up)
                        _xt_down = self._create_excitation(
                            _gs_down.copy(), iocc_down, ivirt_down)
                        cup, cdown = self._append_excitations(
                            cup, cdown, _xt_up, _xt_down)

        # double with 2elec excited on spin up
        for occ1, occ2 in torch.combinations(torch.tensor(idx_occ_up), r=2):
            for vrt1, vrt2 in torch.combinations(torch.tensor(idx_vrt_up), r=2):
                _xt_up = self._create_excitation(
                    _gs_up.copy(), occ1, vrt2)
                _xt_up = self._create_excitation(_xt_up, occ2, vrt1)
                cup, cdown = self._append_excitations(
                    cup, cdown, _xt_up, _gs_down)

        # double with 2elec excited per spin
        for occ1, occ2 in torch.combinations(torch.tensor(idx_occ_down), r=2):
            for vrt1, vrt2 in torch.combinations(torch.tensor(idx_vrt_down), r=2):

                _xt_down = self._create_excitation(
                    _gs_down.copy(), occ1, vrt2)
                _xt_down = self._create_excitation(
                    _xt_down, occ2, vrt1)
                cup, cdown = self._append_excitations(
                    cup, cdown, _gs_up, _xt_down)

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

    def _create_excitation(self, conf, iocc, ivirt):
        return self._create_excitation_replace(conf, iocc, ivirt)

    @staticmethod
    def _create_excitation_ordered(conf, iocc, ivirt):
        """promote an electron from iocc to ivirt

        Args:
            conf (list): index of the occupied orbitals
            iocc (int): index of the occupied orbital
            ivirt (int): index of the virtual orbital

        Returns:
            list: new configuration by increasing order
                  e.g: 4->6 leads to : [0,1,2,3,5,6]
        Note:
            if that method is used to define the exciation index
            permutation must be accounted for when  computing
            the determinant as
            det(A[:,[0,1,2,3]]) = -det(A[:,[0,1,3,2]])
            see : ExcitationMask.get_index_unique_single()
                  in oribtal_projector.py
        """
        conf.pop(iocc)
        conf += [ivirt]
        return conf

    @staticmethod
    def _create_excitation_replace(conf, iocc, ivirt):
        """promote an electron from iocc to ivirt

        Args:
            conf (list): index of the occupied orbitals
            iocc (int): index of the occupied orbital
            ivirt (int): index of the virtual orbital

        Returns:
            list: new configuration not ordered
                e.g.: 4->6 leads tpo : [0,1,2,3,6,5]
        """
        conf[iocc] = ivirt
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


def get_excitation(configs):
    """get the excitation data

    Args:
        configs (tuple): configuratin of the electrons

    Returns:
        exc_up, exc_down : index of the obitals in the excitaitons
                            [i,j],[l,m] : excitation i -> l, j -> l
    """
    exc_up, exc_down = [], []
    for ic, (cup, cdown) in enumerate(zip(configs[0], configs[1])):

        set_cup = set(tuple(cup.tolist()))
        set_cdown = set(tuple(cdown.tolist()))

        if ic == 0:
            set_gs_up = set_cup
            set_gs_down = set_cdown

        else:
            exc_up.append([list(set_gs_up.difference(set_cup)),
                           list(set_cup.difference(set_gs_up))])

            exc_down.append([list(set_gs_down.difference(set_cdown)),
                             list(set_cdown.difference(set_gs_down))])

    return (exc_up, exc_down)


def get_unique_excitation(configs):
    """get the unique excitation data

    Args:
        configs (tuple): configuratin of the electrons

    Returns:
        exc_up, exc_down : index of the obitals in the excitaitons
                            [i,j],[l,m] : excitation i -> l, j -> l
        index_up, index_down : index map for the unique exc
                                [0,0,...], [0,1,...] means that
                                1st : excitation is composed of unique_up[0]*unique_down[0]
                                2nd : excitation is composed of unique_up[0]*unique_down[1]
                                ....

    """
    uniq_exc_up, uniq_exc_down = [], []
    index_uniq_exc_up, index_uniq_exc_down = [], []
    for ic, (cup, cdown) in enumerate(zip(configs[0], configs[1])):

        set_cup = set(tuple(cup.tolist()))
        set_cdown = set(tuple(cdown.tolist()))

        if ic == 0:
            set_gs_up = set_cup
            set_gs_down = set_cdown

        exc_up = [list(set_gs_up.difference(set_cup)),
                  list(set_cup.difference(set_gs_up))]

        exc_down = [list(set_gs_down.difference(set_cdown)),
                    list(set_cdown.difference(set_gs_down))]

        if exc_up not in uniq_exc_up:
            uniq_exc_up.append(exc_up)

        if exc_down not in uniq_exc_down:
            uniq_exc_down.append(exc_down)

        index_uniq_exc_up.append(uniq_exc_up.index(exc_up))
        index_uniq_exc_down.append(
            uniq_exc_down.index(exc_down))

    return (uniq_exc_up, uniq_exc_down), (index_uniq_exc_up, index_uniq_exc_down)
