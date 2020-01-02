import torch


class OrbitalProjector(object):

    def __init__(self, configs, mol):

        self.configs = configs
        self.nconfs = len(configs[0])
        self.nmo = mol.norb
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.Pup, self.Pdown = self.get_projectors()

    def get_projectors(self):
        """Get the projectors of the conf in the CI expansion

        Returns:
            torch.tensor, torch.tensor : projectors
        """

        Pup = torch.zeros(self.nconfs, self.nmo, self.nup)
        Pdown = torch.zeros(self.nconfs, self.nmo, self.ndown)

        for ic, (cup, cdown) in enumerate(zip(self.configs[0], self.configs[1])):

            for _id, imo in enumerate(cup):
                Pup[ic][imo, _id] = 1.

            for _id, imo in enumerate(cdown):
                Pdown[ic][imo, _id] = 1.

        return Pup.unsqueeze(1), Pdown.unsqueeze(1)

    def split_orbitals(self, mo):
        return mo[:, :self.nup, :] @ self.Pup, mo[:, self.nup:, :] @ self.Pdown


class SingletExcitationProjector(object):

    def __init__(self, configs, mol):

        self.configs = configs
        self.nconfs = len(configs[0])
        self.nmo = mol.norb
        self.nup = mol.nup
        self.ndown = mol.ndown

        self.Pup, self.Pdown = self.get_ground_state_projector()
        self.Ext_up_occ, self.Ext_up_virt, \
            self.Ext_down_occ, self.Ext_down_virt  \
            = self.get_singlet_projectors()

    def get_ground_state_projector(self):

        Pup = torch.zeros(self.nmo, self.nup)
        Pdown = torch.zeros(self.nmo, self.ndown)

        cup = self.configs[0][0]
        cdown = self.configs[1][0]

        for _id, imo in enumerate(cup):
            Pup[imo, _id] = 1.

        for _id, imo in enumerate(cdown):
            Pdown[imo, _id] = 1.

        return Pup.unsqueeze(0), Pdown.unsqueeze(0)

    def get_singlet_projectors(self):
        """Get the projectors of the conf in the CI expansion

        Returns:
            torch.tensor, torch.tensor : projectors
        """

        Pup_occ = torch.zeros(self.nconfs, self.nup, 1)
        Pup_virt = torch.zeros(self.nconfs, self.nmo, 1)

        Pdown_occ = torch.zeros(self.nconfs, self.ndown, 1)
        Pdown_virt = torch.zeros(self.nconfs, self.nmo, 1)

        cup_gs = self.configs[0][0]
        cdown_gs = self.configs[1][0]

        for ic, (cup, cdown) in enumerate(zip(self.configs[0][1:], self.configs[1][1:])):

            _check_up = (cup == cup_gs)
            _idx_up = torch.where(_check_up is False)[0].item()

            Pup_occ[ic, _idx_up] = 1.
            Pup_virt[ic, cup[_idx_up]] = 1.

            _check_down = (cdown == cdown_gs)
            _idx_down = torch.where(_check_down is False)[0].item()

            Pdown_occ[ic, _idx_down] = 1.
            Pdown_virt[ic, cup[_idx_down]] = 1.

        return Pup_occ.unsqueeze(1), Pup_virt.unsqueeze(1), Pdown_occ.unsqueeze(1), Pdown_virt.unsqueeze(1)

    def split_orbitals(self, mo):
        return mo[:, :self.nup, :] @ self.Pup, mo[:, self.nup:, :] @ self.Pdown
