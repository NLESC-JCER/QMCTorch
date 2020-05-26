# This file contains the class creating the single orbital network for the Fermi Net
# using the output from the intermediate layers we can construct the single orbital output for orbital i of determinant k
import torch 
import torch.nn as nn

class Orbital_FermiNet(nn.Module):
    def __init__(self, mol, hidden_nodes_e):
        super().__init__()
        '''Fermi Net single orbital'''
        self.mol = mol
        self.N_nuclei = mol.natom
        N_dim = 3

        # this linear layer gives (w_i^ka h_j^La + g_i^ka)
        self.final_linear = nn.Linear(hidden_nodes_e, 1, bias=True)

        # this linear is used for the pi variable with the exponents as input
        self.linear_anisotropic = nn.Linear(self.N_nuclei, 1, bias=False)

        # this linear layer is used for the exponent term of the anisotropic decay.
        # this has a 3 node output from which the eucledian distance is taken.
        self.anisotropic = nn.ModuleList()
        for i in range(self.N_nuclei):
            self.anisotropic.append(
                nn.Linear(N_dim, N_dim, bias=False))

    def forward(self,h_L_j, r_j):
        '''Forward for electron j:
        The input of the orbital is the one-electron stream and position of electron j and the position of the nuclei.'''
        
        # nuclei positions
        R_Nuclei = torch.tensor(self.mol.atom_coords).clone()

        # the output of the intermediate layers goes through the linear layer (w_i^ka h_j^La + g_i^ka)
        a_h_lin = self.final_linear(h_L_j)
        #the anisotropic exponents are determinent
        # the norm is taken from the 3 element output of the linear anisotropic output and put into the exponential 
        anisotropic_exp = torch.zeros((self.N_nuclei))
        for m in range(self.N_nuclei):
            anisotropic_exp[m] = torch.exp(
                -torch.norm(self.anisotropic[m](r_j-R_Nuclei[m])))
        # the output of the exponents is fed into a linear layer with weights pi.
        a_an_lin = self.linear_anisotropic(anisotropic_exp)
        # the anisotropic term and linear electron term are multiplied
        return a_h_lin*a_an_lin

