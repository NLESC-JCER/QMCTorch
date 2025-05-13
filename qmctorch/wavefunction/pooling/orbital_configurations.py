import torch
from typing import Tuple, List
from ...scf import Molecule


class OrbitalConfigurations:
    def __init__(self, mol: Molecule) -> None:
        self.nup = mol.nup
        self.ndown = mol.ndown
        self.nelec = self.nup + self.ndown
        self.spin = mol.spin
        self.norb = mol.basis.nmo

    def get_configs(self, configs: str) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Get the configurations in the CI expansion.

        Args:
            configs (str): Name of the configs we want.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: The spin up/spin down
            electronic configurations.
        """

        if isinstance(configs, str):
            configs = configs.lower()

        if isinstance(configs, tuple):
            assert len(configs) == 2
            assert configs[0].shape == configs[1].shape
            assert len(configs[0][0]) == self.nup
            assert len(configs[0][0]) == self.ndown
            return configs

        elif configs == "ground_state":
            return self._get_ground_state_config()

        elif configs.startswith("cas("):
            nelec, norb = eval(configs.lstrip("cas"))
            self.sanity_check(nelec, norb)
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_cas_config(nocc, nvirt, nelec)

        elif configs.startswith("single("):
            nelec, norb = eval(configs.lstrip("single"))
            self.sanity_check(nelec, norb)
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_single_config(nocc, nvirt)

        elif configs.startswith("single_double("):
            nelec, norb = eval(configs.lstrip("single_double"))
            self.sanity_check(nelec, norb)
            nocc, nvirt = self._get_orb_number(nelec, norb)
            return self._get_single_double_config(nocc, nvirt)

        else:
            print(configs, " not recognized as valid configuration")
            print("Options are : ground_state")
            print("              single(nelec,norb)")
            print("              single_double(nelec,norb)")
            print("              cas(nelec,norb)")
            print("              tuple(tesnor,tensor)")
            raise ValueError("Config error")

    def sanity_check(self, nelec: int, norb: int) -> None:
        """Check if the number of elec/orb is consistent with the
           properties of the molecule

        Args:
            nelec (int): required number of electrons in config
            norb (int): required number of orb in config
        """
        if nelec > self.nelec:
            raise ValueError("required number of electron in config too large")

        if norb > self.norb:
            raise ValueError("required number of orbitals in config too large")

    def _get_ground_state_config(self) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Return only the ground state configuration.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: The spin up/spin down
            electronic configurations.
        """
        _gs_up = list(range(self.nup))
        _gs_down = list(range(self.ndown))
        cup, cdown = [_gs_up], [_gs_down]
        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_single_config(
        self, nocc: Tuple[int, int], nvirt: Tuple[int, int]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Get the confs of the singlet conformations

        Args:
            nocc (Tuple[int,int]): number of occupied orbitals in the active space
            nvirt (Tuple[int,int]): number of virtual orbitals in the active space

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: The spin up/spin down
            electronic configurations.
        """

        _gs_up = list(range(self.nup))
        _gs_down = list(range(self.ndown))
        cup, cdown = [_gs_up], [_gs_down]

        for iocc in range(self.nup - 1, self.nup - 1 - nocc[0], -1):
            for ivirt in range(self.nup, self.nup + nvirt[0], 1):
                # create an excitation is spin pu
                _xt = self._create_excitation(_gs_up.copy(), iocc, ivirt)

                # append that excitation
                cup, cdown = self._append_excitations(cup, cdown, _xt, _gs_down)

        for iocc in range(self.ndown - 1, self.ndown - 1 - nocc[1], -1):
            for ivirt in range(self.ndown, self.ndown + nvirt[1], 1):
                # create an excitation is spin down
                _xt = self._create_excitation(_gs_down.copy(), iocc, ivirt)

                # append that excitation
                cup, cdown = self._append_excitations(cup, cdown, _gs_up, _xt)

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_single_double_config(
        self, nocc: Tuple[int, int], nvirt: Tuple[int, int]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
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

        idx_occ_up = list(range(self.nup - 1, self.nup - 1 - nocc[0], -1))
        idx_vrt_up = list(range(self.nup, self.nup + nvirt[0], 1))

        idx_occ_down = list(range(self.ndown - 1, self.ndown - 1 - nocc[1], -1))
        idx_vrt_down = list(range(self.ndown, self.ndown + nvirt[1], 1))

        # ground, single and double with 1 elec excited per spin
        for iocc_up in idx_occ_up:
            for ivirt_up in idx_vrt_up:
                for iocc_down in idx_occ_down:
                    for ivirt_down in idx_vrt_down:
                        _xt_up = self._create_excitation(
                            _gs_up.copy(), iocc_up, ivirt_up
                        )
                        _xt_down = self._create_excitation(
                            _gs_down.copy(), iocc_down, ivirt_down
                        )
                        cup, cdown = self._append_excitations(
                            cup, cdown, _xt_up, _xt_down
                        )

        # double with 2elec excited on spin up
        for occ1, occ2 in torch.combinations(torch.as_tensor(idx_occ_up), r=2):
            for vrt1, vrt2 in torch.combinations(torch.as_tensor(idx_vrt_up), r=2):
                _xt_up = self._create_excitation(_gs_up.copy(), occ1, vrt2)
                _xt_up = self._create_excitation(_xt_up, occ2, vrt1)
                cup, cdown = self._append_excitations(cup, cdown, _xt_up, _gs_down)

        # double with 2elec excited per spin
        for occ1, occ2 in torch.combinations(torch.as_tensor(idx_occ_down), r=2):
            for vrt1, vrt2 in torch.combinations(torch.as_tensor(idx_vrt_down), r=2):
                _xt_down = self._create_excitation(_gs_down.copy(), occ1, vrt2)
                _xt_down = self._create_excitation(_xt_down, occ2, vrt1)
                cup, cdown = self._append_excitations(cup, cdown, _gs_up, _xt_down)

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_cas_config(
        self, nocc: Tuple[int, int], nvirt: Tuple[int, int], nelec: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """get confs of the CAS

        Args:
            nocc (int): number of occupied orbitals in the CAS
            nvirt ([type]): number of virt orbitals in the CAS
        """
        from itertools import combinations, product

        if self.spin != 0:
            raise ValueError(
                "CAS active space not possible with spin polarized calculation"
            )

        idx_low, idx_high = self.nup - nocc[0], self.nup + nvirt[0]
        orb_index_up = range(idx_low, idx_high)
        idx_frz = list(range(idx_low))
        _cup = [idx_frz + list(l) for l in list(combinations(orb_index_up, nelec // 2))]

        idx_low, idx_high = self.nup - nocc[0] - 1, self.nup + nvirt[0] - 1

        _cdown = [
            idx_frz + list(l) for l in list(combinations(orb_index_up, nelec // 2))
        ]

        confs = list(product(_cup, _cdown))
        cup, cdown = [], []

        for c in confs:
            cup.append(c[0])
            cdown.append(c[1])

        return (torch.LongTensor(cup), torch.LongTensor(cdown))

    def _get_orb_number(
        self, nelec: int, norb: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """compute the number of occupied and virtual orbital
        __ PER SPIN __
        __ ONLY VALID For spin up/down ___
        Args:
            nelec (int): total number of elec in the CAS
            norb (int): total number of orb in the CAS

        Returns:
            [int,int]: number of occpuied/virtual orb per spin
        """

        # determine the number of occupied mo per spin in the active space
        if nelec % 2 == 0:
            nocc = (nelec // 2, nelec // 2)
        else:
            nocc = (nelec // 2 + 1, nelec // 2)

        # determine the number of virt mo per spin in the active space
        nvirt = (norb - nocc[0], norb - nocc[1])
        return nocc, nvirt

    def _create_excitation(self, conf: List[int], iocc: int, ivirt: int) -> List[int]:
        """promote an electron from iocc to ivirt

        Args:
            conf (list): index of the occupied orbitals
            iocc (int): index of the occupied orbital
            ivirt (int): index of the virtual orbital

        Returns:
            list: new configuration by replacing the iocc index with ivirt
        """
        return self._create_excitation_replace(conf, iocc, ivirt)

    @staticmethod
    def _create_excitation_ordered(conf: List[int], iocc: int, ivirt: int) -> List[int]:
        """promote an electron from iocc to ivirt

        Args:
            conf (List[int]): index of the occupied orbitals
            iocc (int): index of the occupied orbital
            ivirt (int): index of the virtual orbital

        Returns:
            List[int]: new configuration by increasing order
                       e.g: 4->6 leads to : [0,1,2,3,5,6]

        Note:
            if that method is used to define the exciation index
            permutation must be accounted for when  computing
            the determinant as
            det(A[:,[0,1,2,3]]) = -det(A[:,[0,1,3,2]])
            see : ExcitationMask.get_index_unique_single()
                  in oribtal_projector.py
        """

    @staticmethod
    def _create_excitation_replace(conf: List[int], iocc: int, ivirt: int) -> List[int]:
        """promote an electron from iocc to ivirt

        Args:
            conf (List[int]): index of the occupied orbitals
            iocc (int): index of the occupied orbital
            ivirt (int): index of the virtual orbital

        Returns:
            List[int]: new configuration not ordered
                e.g.: 4->6 leads tpo : [0,1,2,3,6,5]
        """
        conf[iocc] = ivirt
        return conf

    @staticmethod
    def _append_excitations(
        cup: List[List[int]],
        cdown: List[List[int]],
        new_cup: List[int],
        new_cdown: List[int],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Append new excitations

        Args:
            cup: configurations of spin up
            cdown: configurations of spin down
            new_cup: new spin up confs
            new_cdown: new spin down confs

        Returns:
            cup: updated list of spin up confs
            cdown: updated list of spin down confs
        """
        cup.append(new_cup)
        cdown.append(new_cdown)
        return cup, cdown


def get_excitation(
    configs: Tuple[torch.LongTensor, torch.LongTensor]
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    """Get the excitation data

    Args:
        configs: tuple of two tensors of shape (nconfig, norb)
            configuratin of the electrons

    Returns:
        exc_up, exc_down : two lists of lists of lists of integers
            excitation i -> l, j -> l
            exc_up[i][0] : occupied orbital, exc_up[i][1] : virtual orbital
            exc_down[i][0] : occupied orbital, exc_down[i][1] : virtual orbital
    """
    exc_up, exc_down = [], []
    for ic, (cup, cdown) in enumerate(zip(configs[0], configs[1])):
        set_cup = set(tuple(cup.tolist()))
        set_cdown = set(tuple(cdown.tolist()))

        if ic == 0:
            set_gs_up = set_cup
            set_gs_down = set_cdown

        else:
            exc_up.append(
                [
                    list(set_gs_up.difference(set_cup)),
                    list(set_cup.difference(set_gs_up)),
                ]
            )

            exc_down.append(
                [
                    list(set_gs_down.difference(set_cdown)),
                    list(set_cdown.difference(set_gs_down)),
                ]
            )

    return (exc_up, exc_down)


def get_unique_excitation(
    configs: Tuple[torch.LongTensor, torch.LongTensor]
) -> Tuple[Tuple[List[List[int]], List[List[int]]], Tuple[List[int], List[int]]]:
    """get the unique excitation data

    Args:
        configs (tuple): configuratin of the electrons

    Returns:
        uniq_exc (tuple): unique excitation data
            uniq_exc[0] (list): unique excitation of spin up
            uniq_exc[1] (list): unique excitation of spin down
        index_uniq_exc (tuple): index map for the unique exc
            index_uniq_exc[0] (list): index of the unique excitation of spin up
            index_uniq_exc[1] (list): index of the unique excitation of spin down
    """
    uniq_exc_up, uniq_exc_down = [], []
    index_uniq_exc_up, index_uniq_exc_down = [], []
    for ic, (cup, cdown) in enumerate(zip(configs[0], configs[1])):
        set_cup = set(tuple(cup.tolist()))
        set_cdown = set(tuple(cdown.tolist()))

        if ic == 0:
            set_gs_up = set_cup
            set_gs_down = set_cdown

        exc_up = [
            list(set_gs_up.difference(set_cup)),
            list(set_cup.difference(set_gs_up)),
        ]

        exc_down = [
            list(set_gs_down.difference(set_cdown)),
            list(set_cdown.difference(set_gs_down)),
        ]

        if exc_up not in uniq_exc_up:
            uniq_exc_up.append(exc_up)

        if exc_down not in uniq_exc_down:
            uniq_exc_down.append(exc_down)

        index_uniq_exc_up.append(uniq_exc_up.index(exc_up))
        index_uniq_exc_down.append(uniq_exc_down.index(exc_down))

    return ((uniq_exc_up, uniq_exc_down), (index_uniq_exc_up, index_uniq_exc_down))
