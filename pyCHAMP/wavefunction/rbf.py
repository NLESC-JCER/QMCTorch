import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x,y):

    xn = (x**2).sum(1).view(-1,1)
    yn = (y**2).sum(1).view(1,-1)
    return xn + yn 2.*x.mm(y.transpose(0,1))


class RBF1D(nn.Module):

    def __init__(self,output_features,centers,opt_centers=True):
        '''Radial Basis Function Layer in 1D

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
        '''

        super(RBF,self).__init__()
        #self.input_features = 1
        self.output_features = output_features

        self.centers = nn.Parameter(torch.Tensor(centers))
        self.centers.requires_grad = opt_centers

        self.sigma = self.get_sigma(self.centers)

        self.weight = nn.Parameter(torch.Tensor(output_features,1))
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
        

class RBF2D(nn.Module):

        def __init__(self,output_features,centers,opt_centers=True):
        '''Radial Basis Function Layer in 1D

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
        '''

        super(RBF,self).__init__()
        self.output_features = output_features

        self.centers = nn.Parameter(torch.Tensor(centers))
        self.centers.requires_grad = opt_centers

        self.sigma = self.get_sigma(self.centers)

        self.weight = nn.Parameter(torch.Tensor(output_features,2))
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
        out = torch.exp( - pairwise_distance(out,self.centers) / self.sigma )
        return(out)