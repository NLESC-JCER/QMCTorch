import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI

def pairwise_distance(x,y):

    xn = (x**2).sum(1).view(-1,1)
    yn = (y**2).sum(1).view(1,-1)
    return xn + yn + 2.*x.mm(y.transpose(0,1))


class RBF1D(nn.Module):

    def __init__(self,input_features,output_features,centers,opt_centers=True):
        '''Radial Basis Function Layer in 1D

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
        '''

        super(RBF1D,self).__init__()
        self.input_features = input_features
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
        

class RBF(nn.Module):

    def __init__(self,input_features,output_features,centers,opt_centers=True):
        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
        '''

        super(RBF,self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(torch.Tensor(centers))
        self.centers.requires_grad = opt_centers

        # get the standard deviations
        self.sigma = self.get_sigma(self.centers)
        #self.sigma = self.get_sigma_ones(self.centers)

        # get the covariance matrix and its inverse
        self.cov = self.covmat(self.sigma)
        self.invCov = torch.inverse(self.cov) 

        # get the scaled determinant of the cov matrices
        self.detS = (self.sigma**2).sum(1).view(-1,1)
        k = (2.*PI)**self.input_features
        self.detS = torch.sqrt( k*self.detS )

    @staticmethod
    def get_sigma(X):
        npoints = torch.tensor(float(len(X)))
        nsqrt = torch.sqrt(npoints) - 1.
        delta = 0.5*(X.max(0).values - X.min(0).values) / nsqrt
        return delta.expand(X.size())

    @staticmethod
    def get_sigma_ones(X):
        return torch.ones(X.shape)

    @staticmethod
    def covmat(sigma):
        s2 = sigma**2
        I = torch.eye(sigma.size(1))
        cov = s2.unsqueeze(2).expand(*s2.size(),s2.size(1))
        return cov * I
        
    def forward(self,input):
        '''Compute the output of the RBF layer'''

        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta =  input[:,None,:] - self.centers[None,...]

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = ( torch.matmul(delta.unsqueeze(2),self.invCov).squeeze() * delta ).sum(2)

        # divide by the determinant of the cov mat
        X = (torch.exp(-0.5*X).unsqueeze(2) / self.detS).squeeze()

        return X