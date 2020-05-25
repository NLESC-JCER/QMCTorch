import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import math
from numpy import sqrt, exp, sum, random
from numpy.linalg import norm

CUDA_LAUNCH_BLOCKING = 1
use_cuda = True

if use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device

def f_concatenate(h_i, h_ij, N_spin):
    '''Function to concatenate the desired information to get the input for each new layer.
    With as input the one electron and two electron stream output of the previous layer and
    the spin assignment of the electrons.'''
    g_down = torch.mean(h_i[:N_spin[0]], axis=0).repeat(h_i.shape[0], 1)
    g_up = torch.mean(h_i[N_spin[0]:], axis=0).repeat(h_i.shape[0], 1)
    g_down_i = torch.mean(h_ij[:N_spin[0]], axis=0)
    g_up_i = torch.mean(h_ij[N_spin[0]:], axis=0)
    f_i = torch.cat((h_i, g_down, g_up, g_down_i, g_up_i), axis=1)
    # outputs a array f_i where the first dimension are the electrons i
    return f_i


def f_size(hidden_nodes_e, hidden_nodes_ee):
    '''Function to get the input size for the hidden layers'''
    return 3*hidden_nodes_e + 2*(hidden_nodes_ee)


def electron_nuclei_input(r_electrons, R_Nuclei):
    '''Function to create intial input of electron-nuclei distances.'''
    # input of electron and nuclei positions
    h_0_i = torch.tensor([])
    # measure the distances between electrons and the nuclei,
    # and determine the input for the single electron stream
    for l in range(R_Nuclei.shape[0]):
        r_il = torch.tensor(r_electrons-R_Nuclei[l])
        r_il_len = torch.norm(r_il, dim=1).reshape(r_electrons.shape[0], 1)
        h_0_il = torch.cat((r_il, r_il_len), axis=1)
        h_0_i = torch.cat(
            (h_0_i, h_0_il), axis=1) if h_0_i.size else h_0_il
    return h_0_i


def electron_electron_input(r_electrons):
    '''Function to create intial input of electron-electron distances.'''
    # input of electron positions

    # create meshgrid for indexing of electron i and j
    [i, j] = torch.meshgrid(torch.arange(
        0, r_electrons.shape[0]), torch.arange(0, r_electrons.shape[0]))
    xij = torch.zeros((r_electrons.shape[0], r_electrons.shape[0], 3))
    # determine electron - electron distance vector
    xij[:, :, :] = torch.tensor(
        r_electrons[i, :]-r_electrons[j, :]).reshape(r_electrons.shape[0], r_electrons.shape[0], 3)
    # determine absolute distance
    rij = torch.norm(xij, dim=2)
    h_0_ij = torch.cat(
        (xij, rij.reshape(r_electrons.shape[0], r_electrons.shape[0], 1)), axis=2)

    return h_0_ij


class Intermediate_Layers_v1(nn.Module):
    def __init__(self, N_electrons, N_nuclei, N_spin, hidden_nodes_e, hidden_nodes_ee, L_layers):
        super().__init__()
        '''Intermediate layer where each one-electron stream has a iddentical weight matrix V_l
        '''

        self.N_nuclei = N_nuclei
        self.N_electrons = N_electrons
        self.N_spin = N_spin
        self.L_layers = L_layers

        N_dim = 3
        # determine the initial input size  based on the concatenation
        input_size = (3*N_nuclei+2)*(N_dim+1)

        # linear input layer
        # Here it is assumed that each input i has the same weights V_l
        # nn.ModuleList() creates a list object to which nearal network layers can be appended as seperate layers.
        self.lin_layer_e = nn.ModuleList()
        self.lin_layer_e.append(
            nn.Linear(input_size, hidden_nodes_e, bias=True))
        for l in range(1, self.L_layers):
            self.lin_layer_e.append(
                nn.Linear(f_size(hidden_nodes_e, hidden_nodes_ee), hidden_nodes_e, bias=True))

        self.lin_layer_ee = nn.ModuleList()
        self.lin_layer_ee.append(
            nn.Linear(N_dim+1, hidden_nodes_ee, bias=True))
        for l in range(1, self.L_layers):
            self.lin_layer_ee.append(
                nn.Linear(hidden_nodes_ee, hidden_nodes_ee, bias=True))

    def forward(self, r_electrons, R_Nuclei):
        # input of electron and nuclei positions

        # Look at the electron nuclei distances
        h_i = electron_nuclei_input(r_electrons, R_Nuclei)

        # Now look at the electron electron distances
        h_ij = electron_electron_input(r_electrons)

        # now that we have the desired interaction distances we will concatenate to f

        for l in range(self.L_layers):
            # for the one electron stream:
            h_i_previous = h_i
            f = f_concatenate(h_i, h_ij, self.N_spin)
            l_e = self.lin_layer_e[l](f)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_i = torch.tanh(l_e)
            if h_i.shape[1] == h_i_previous.shape[1]:
                h_i = h_i + h_i_previous
            # for the two electron stream:
            h_ij_previous = h_ij
            l_ee = self.lin_layer_ee[l](h_ij)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_ij = torch.tanh(l_ee)
            if h_ij.shape[2] == h_ij_previous.shape[2]:
                h_ij = h_ij + h_ij_previous

        return h_i, h_ij
    


class one_electron_layer(nn.Module):
    def __init__(self, input_size, hidden_nodes, N_electrons):
        super().__init__()
        self.hidden_nodes = hidden_nodes
        '''Create a concatenation of linear layers for each electron i'''
        self.list_linear = nn.ModuleList()
        for i in range(N_electrons):
            self.list_linear.append(
                nn.Linear(input_size, self.hidden_nodes, bias=True))

    def forward(self, f_i):

        h_i = torch.zeros((f_i.shape[0], self.hidden_nodes))
        for i in range(f_i.shape[0]):
            h_i[i] = self.list_linear[i](f_i[i])

        return h_i


# class two_electron_layer(nn.Module):
#     def __init__(self, input_size, hidden_nodes, N_electrons):
#         super().__init__()
#         self.hidden_nodes = hidden_nodes
#         '''Create a concatenation of linear layers for each electron i'''
#         self.list_linear = nn.ModuleList()
#         for i in range(N_electrons^2):
#             self.list_linear.append(
#                 nn.Linear(input_size, self.hidden_nodes, bias=True))

#     def forward(self, h_ij):

#         h_ij = torch.zeros((h_ij.shape[0], h_ij.shape[1],self.hidden_nodes))
#         for i in range(h_ij.shape[0]):
#             h_i[i] = self.list_linear[i](h_i[i])

#         return h_i


class Intermediate_Layers_v2(nn.Module):
    def __init__(self, N_electrons, N_nuclei, N_spin, hidden_nodes_e, hidden_nodes_ee, L_layers):
        super().__init__()
        '''Fermi Net intermediate layers'''

        self.N_nuclei = N_nuclei
        self.N_electrons = N_electrons
        self.N_spin = N_spin
        self.L_layers = L_layers

        N_dim = 3
        input_size = (3*N_nuclei+2)*(N_dim+1)

        # linear input layer
        # Here it is assumed that each electron input i has a different weights V_l      
        # we use the one_electron_layer function defined above to create seperate 
        # linear connections for each electron
        self.lin_layer_e = nn.ModuleList()
        self.lin_layer_e.append(one_electron_layer(
            input_size, hidden_nodes_e, N_electrons))
        for l in range(1, self.L_layers):
            self.lin_layer_e.append(one_electron_layer(
                f_size(hidden_nodes_e, hidden_nodes_ee), hidden_nodes_e, N_electrons))

        self.lin_layer_ee = nn.ModuleList()
        self.lin_layer_ee.append(
            nn.Linear(N_dim+1, hidden_nodes_ee, bias=True))
        for l in range(1, self.L_layers):
            self.lin_layer_ee.append(
                nn.Linear(hidden_nodes_ee, hidden_nodes_ee, bias=True))
    
    def forward(self, r_electrons, R_Nuclei):
        # input of electron and nuclei positions

        # Look at the electron nuclei distances
        h_i = electron_nuclei_input(r_electrons, R_Nuclei)

        # Now look at the electron electron distances
        h_ij = electron_electron_input(r_electrons)

        # now that we have the desired interaction distances we will concatenate to f

        for l in range(self.L_layers):
            # for the one electron stream:
            h_i_previous = h_i
            f = f_concatenate(h_i, h_ij, self.N_spin)
            l_e = self.lin_layer_e[l](f)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_i = torch.tanh(l_e)
            if h_i.shape[1] == h_i_previous.shape[1]:
                h_i = h_i + h_i_previous
            # for the two electron stream:
            h_ij_previous = h_ij
            l_ee = self.lin_layer_ee[l](h_ij)
            # with a tanh activation and dependent on the hidden layers size a residual connection
            h_ij = torch.tanh(l_ee)
            if h_ij.shape[2] == h_ij_previous.shape[2]:
                h_ij = h_ij + h_ij_previous

        return h_i, h_ij
   


def getNumParams(params):
    '''function to get the variable count'''
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

#from https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/12


# test the intermediate layers .
# for this initial example I will start with a smaller 4 particle system with 2 spin up and 2 spin down particles.
N_spin = [2, 2]
N_electrons = np.sum(N_spin)
# with a system in 3 dimensional space
N_dim = 3
# with 1 nuclei
N_nuclei = 1

# network hyperparameters:
N_intermediate_layers = 2 
hidden_nodes_e = 256
hidden_nodes_ee = 32
K_determinants = 1
L_layers = 4

# set a initial seed for to make the example reproducable
torch.random.manual_seed(321)

# initiaite a random configuration of particle positions
r = torch.randn(N_electrons, N_dim, device="cpu")
# r = torch.ones(N_electrons, N_dim, device="cpu")

# and for the nuclei
R = torch.randn(N_nuclei, N_dim, device="cpu")

FN = Intermediate_Layers_v1(N_electrons, N_nuclei,
                            N_spin, hidden_nodes_e, hidden_nodes_ee, L_layers)

# check the number of parameters and layers of the intermediate layers:

for name, param in FN.named_parameters():
    print(name, param.size())

print(getNumParams(FN.parameters()))

# check the output of the network:

# h_i, h_ij = FN.forward(r, R)
# print("h_i is given by: \n {}".format(h_i))
# print("h_ij is outputted as:\n {}".format(h_ij))


class orbital_k_i(nn.Module):
    def __init__(self, N_nuclei, intermediate_nodes_e):
        super().__init__()
        '''Fermi Net single orbital'''
        self.N_nuclei = N_nuclei
        N_dim = 3

        # this linear layer gives (w_i^ka h_j^La + g_i^ka)
        self.final_linear = nn.Linear(intermediate_nodes_e, 1, bias=True)

        # this linear is used for the pi variable with the exponents as input
        self.linear_anisotropic = nn.Linear(N_nuclei, 1, bias=False)

        # this linear layer is used for the exponent term of the anisotropic decay.
        # this has a 3 node output from which the eucledian distance is taken.
        self.anisotropic = nn.ModuleList()
        for i in range(N_nuclei):
            self.anisotropic.append(
                nn.Linear(N_dim, N_dim, bias=False))

    def forward(self,h_L_j, r_j, R):
        '''Forward for electron j:
        The input of the orbital is the one-electron stream and position of electron j and the position of the nuclei.'''
        
        # the output of the intermediate layers goes through the linear layer (w_i^ka h_j^La + g_i^ka)
        a_h_lin = self.final_linear(h_L_j)
        #the anisotropic exponents are determinent
        # the norm is taken from the 3 element output of the linear anisotropic output and put into the exponential 
        anisotropic_exp = torch.zeros((self.N_nuclei))
        for m in range(self.N_nuclei):
            anisotropic_exp[m] = torch.exp(
                -torch.norm(self.anisotropic[m](r_j-R[m])))
        # the output of the exponents is fed into a linear layer with weights pi.
        a_an_lin = self.linear_anisotropic(anisotropic_exp)
        # the anisotropic term and linear electron term are multiplied
        return a_h_lin*a_an_lin


# this function gives the output for a single orbital and can be used in the determinant 

# create the network:
N_spin = [2, 2]
N_electrons = np.sum(N_spin)
# with a system in 3 dimensional space
N_dim = 3
# with 1 nuclei
N_nuclei = 1

# network hyperparameters:
N_intermediate_layers = 4
hidden_nodes_e = 256
hidden_nodes_ee = 32
K_determinants = 1

FN = Intermediate_Layers_v1(N_electrons, N_nuclei,
                            N_spin, hidden_nodes_e, hidden_nodes_ee,N_intermediate_layers)
OR = orbital_k_i(N_nuclei, hidden_nodes_e)

# check the number of parameters and layers of the network:
for name, param in OR.named_parameters():
    print(name, param.size())

print(getNumParams(OR.parameters()))

# check the output of the network:
h_i, h_ij = FN.forward(r, R)
print("h_i is given by: \n {}".format(h_i))
# and the output for the orbital
orbital = OR.forward(h_i[0], r[0], R)
print(orbital)


class Fermi_net(nn.Module):
    def __init__(self, N_electrons, N_nuclei, N_spin, hidden_nodes_e, hidden_nodes_ee, L_layers, K_determinants):
        super().__init__()
        '''Fermi Net wavefunction'''
        self.N_nuclei = N_nuclei
        self.N_electrons = N_electrons
        self.N_spin = N_spin
        self.hidden_nodes_e = hidden_nodes_e
        self.hidden_nodes_ee = hidden_nodes_ee
        self.K_determinants = K_determinants
        self.L_layers = L_layers
        N_dim = 3
        
        # Create the intermediate layer network:
        self.Intermediate_layer = Intermediate_Layers_v1(
            self.N_electrons, self.N_nuclei, self.N_spin, hidden_nodes_e, hidden_nodes_ee, L_layers)
        # create K*N_up "spin up" orbitals and K*N_down "spin down" orbitals with K the number of determinants.
        self.Orbital_determinant_up = nn.ModuleList()
        self.Orbital_determinant_down = nn.ModuleList()
        for k in range(K_determinants):
            Orbitals_up = nn.ModuleList()
            Orbitals_down = nn.ModuleList()
            for i in range(N_spin[0]):
                Orbitals_up.append(orbital_k_i(N_nuclei, hidden_nodes_e))

            for i in range(N_spin[1]):
                Orbitals_down.append(orbital_k_i(N_nuclei, hidden_nodes_e))

            self.Orbital_determinant_up.append(Orbitals_up)
            self.Orbital_determinant_down.append(Orbitals_down)
        # Create a final linear layer of weighted determinants.
        self.weighted_sum = nn.Linear(K_determinants, 1, bias=False)

    def forward(self, r_electrons, R_Nuclei):
        
        # Go through the intermediate layers 
        h_L_i, h_L_ij = self.Intermediate_layer(r_electrons, R_Nuclei)
        
        # Create the determinants 
        determinant = torch.zeros(self.K_determinants)
        for k in range(self.K_determinants):
            # for each determinant k create two slatter determinants one spin-up one spin-down
            det_up = torch.zeros((N_spin[0], N_spin[0]))
            det_down = torch.zeros((N_spin[1], N_spin[1]))
            # for up:
            for i_up in range(N_spin[0]):
                for j_up in range(N_spin[0]):
                    det_up[i_up, j_up] = self.Orbital_determinant_up[k][i_up](
                        h_L_i[j_up], r_electrons[j_up], R_Nuclei)
            # for down:        
            for i_down in range(N_spin[1]):
                for j_down in range(N_spin[1]):
                    j_down = j_down + N_spin[0]
                    det_down[i_down, j_down-N_spin[0]] = self.Orbital_determinant_up[k][i_down](
                        h_L_i[j_down], r_electrons[j_down], R_Nuclei)
            determinant[k] = torch.det(det_up)*torch.det(det_down)
        #create make a weighted sum of the determinants
        psi = self.weighted_sum(determinant)
        # return psi.
        # in the optimization log(psi) is used as output of the network
        # return torch.log(psi)
        return psi


# for this initial example I will start with a smaller 4 particle system with 2 spin up and 2 spin down particles.
N_spin = [2, 2]
N_electrons = np.sum(N_spin)
# with a system in 3 dimensional space
N_dim = 3
# with 1 nuclei
N_nuclei = 1

# network hyperparameters:
N_intermediate_layers = 4
hidden_nodes_e = 256
hidden_nodes_ee = 32
K_determinants = 1

# set a initial seed for to make the example reproducable
torch.random.manual_seed(321)

# initiaite a random configuration of particle positions
r = torch.randn(N_electrons, N_dim, device="cpu")
# r = torch.ones(N_electrons, N_dim, device="cpu")

# and for the nuclei
R = torch.randn(N_nuclei, N_dim, device="cpu")

WF = Fermi_net(N_electrons, N_nuclei, N_spin, hidden_nodes_e,
               hidden_nodes_ee, N_intermediate_layers, K_determinants)

# check the number of parameters and layers of the intermediate layers:

for name, param in WF.named_parameters():
    print(name, param.size())

print(getNumParams(WF.parameters()))

# check the output of the network:
psi = WF.forward(r, R)
print(psi)

