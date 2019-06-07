import torch
from torch import nn
import torch.nn.functional as F
class RBF(nn.Module):

    def __init__(self,input_features,output_features,centers,opt_centers=True):
        '''Radial Basis Function Layer

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
        '''

        super(RBF,self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.centers = nn.Parameter(torch.Tensor(centers))
        self.centers.requires_grad = opt_centers

        self.sigma = self.get_sigma(self.centers)

        self.weight = nn.Parameter(torch.Tensor(output_features,input_features))
        self.weight.data.fill_(1.)
        self.weight.requires_grad = False

        self.register_parameter('bias',None)

    @staticmethod
    def get_sigma(X):
        x = X.clone().detach()
        xp = torch.cat((torch.tensor([x[1]]),x[:-1]))
        xm = torch.cat((x[1:],torch.tensor([x[-2]])))
        return 0.5*(torch.abs(x-xp) + torch.abs(x-xm))

    def forward(self,input):
        '''Compute the output of the RBF layer'''
        self.sigma = self.get_sigma(self.centers)
        out = F.linear(input,self.weight,self.bias)
        out = torch.exp( -(out-self.centers)**2 / self.sigma )
        return(out)
        