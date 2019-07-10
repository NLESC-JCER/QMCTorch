import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI

def pairwise_distance(x,y):

    xn = (x**2).sum(1).view(-1,1)
    yn = (y**2).sum(1).view(1,-1)
    return xn + yn + 2.*x.mm(y.transpose(0,1))

########################################################################################################

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
        
########################################################################################################

class RBF(nn.Module):

    def __init__(self,
                input_features,
                output_features,
                centers,
                kernel='gaussian',
                opt_centers=True,
                sigma = 1.0,
                opt_sigma= False ):

        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF,self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(torch.Tensor(centers))
        self.ncenter = len(self.centers)
        self.centers.requires_grad = opt_centers
        self.kernel = kernel

        # get the standard deviations
        self.sigma_method = sigma
        self.sigma = nn.Parameter(self.get_sigma())
        self.sigma.requires_grad = opt_sigma

        # get the covariance matrix and its inverse
        self.invCov = self.invCovMat(self.sigma)

        # GET THE DENOMINATOR
        self.detS = self.denom(self.sigma,self.input_features)

        # get the scaled determinant of the cov matrices
        # self.detS = (self.sigma**2).prod(1).view(-1,1)
        # k = (2.*PI)**self.input_features
        # self.detS = torch.sqrt( k*self.detS )

    def get_sigma(self):

        if isinstance(self.sigma_method,float):
            return self.get_sigma_ones(self.centers, s=self.sigma_method)
        elif isinstance(self.sigma_method,torch.Tensor):
            return self.sigma_method
        elif self.sigma_method == '1d':
            return self.get_sigma_1d(self.centers)
        elif self.sigma_method == 'mean':
            return self.get_sigma_average(self.centers)
        else:
            raise ValueError(self.sigma_method, ' not a correct option for sigma')

    @staticmethod
    def get_sigma_average(X):
        npoints = torch.tensor(float(len(X)))
        nsqrt = npoints**(1./X.shape[1]) - 1.
        delta = (X.max(0).values - X.min(0).values) / nsqrt
        return delta.expand(X.size())


    @staticmethod
    def get_sigma_1d(X):
        x = X.clone().detach().view(-1)
        xp = torch.cat((torch.tensor([x[1]]),x[:-1]))
        xm = torch.cat((x[1:],torch.tensor([x[-2]])))
        return 0.5*(torch.abs(x-xp) + torch.abs(x-xm)).view(-1,1)

    @staticmethod
    def get_sigma_ones(X,s=1):
        return s*torch.ones(X.shape)

    @staticmethod
    def invCovMat(sigma):
        s2 = sigma**2
        I = torch.eye(sigma.size(1))
        cov = s2.unsqueeze(2).expand(*s2.size(),s2.size(1))
        return torch.inverse(cov * I)

    @staticmethod
    def denom(sigma,dim):
        out = (sigma**2).prod(1).view(-1,1)
        k = (2.*PI)**dim
        return torch.sqrt( k*out )
        
    def forward(self,input):
        '''Compute the output of the RBF layer'''

        if self.kernel == 'gaussian':
            return self._gaussian_kernel(input)
        elif self.kernel == 'slater':
            return self._slater_kernel(input)
        else:
            raise ValueError('Kernel not recognized')

    def _gaussian_kernel(self,input):

        if self.sigma.requires_grad:
            self.invCov = self.invCovMat(self.sigma)
            self.detS = self.denom(self.sigma,self.input_features)

        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta =  (input[:,None,:] - self.centers[None,...])

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = ( torch.matmul(delta.unsqueeze(2),self.invCov).squeeze(2) * delta ).sum(2)

        # slater kernel
        if self.kernel == 'slater':
            X = torch.sqrt(X)
            self.detS[:,:] = 1.

        # divide by the determinant of the cov mat
        X = (torch.exp(-0.5*X).unsqueeze(2) / self.detS).squeeze()

        return X.view(-1,self.ncenter)

    def _slater_kernel(self,input):
        
    
        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta =  (input[:,None,:] - self.centers[None,...])

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = ( torch.matmul(delta.unsqueeze(2),self.invCov).squeeze(2) * delta ).sum(2)
        X = (delta**2).sum(2)
        X = torch.sqrt(X)
        
        # divide by the determinant of the cov mat
        X = torch.exp(-self.sigma_method*X)

        return X.view(-1,self.ncenter)

########################################################################################################

class RBF_Slater(nn.Module):

    def __init__(self,
                input_features,
                output_features,
                centers,
                sigma ):

        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF_Slater,self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(centers)
        self.centers.requires_grad = True
        self.ncenter = len(self.centers)
        
        # get the standard deviations
        self.sigma = nn.Parameter(sigma)
        self.sigma.requires_grad = True

    def forward(self,input):
        
    
        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta =  (input[:,None,:] - self.centers[None,...])

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = (delta**2).sum(2)
        X = torch.sqrt(X)
        
        # divide by the determinant of the cov mat
        X = torch.exp(-self.sigma*X)

        return X.view(-1,self.ncenter)


################################################################################

class RBF_Slater_NELEC(nn.Module):

    def __init__(self,
                input_features,
                output_features,
                centers,
                sigma,
                nelec):

        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF_Slater_NELEC,self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(centers)
        self.centers.requires_grad = True
        self.ncenter = len(self.centers)
        
        # get the standard deviations
        self.sigma = nn.Parameter(sigma)
        self.sigma.requires_grad = True

        # wavefunction data
        self.nelec = nelec
        self.ndim = int(self.input_features/self.nelec)

    def forward(self,input):
        
        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nrbf,Ndim)
        delta =  (input.view(-1,self.nelec,1,self.ndim) - self.centers[None,...])

        # compute the distance
        # -> (Nbatch,Nelec,Nrbf)
        X = (delta**2).sum(3)
        X = torch.sqrt(X)
        
        # multiply by the exponent and take the exponential
        # -> (Nbatch,Nelec,Nrbf)
        X = torch.exp(-self.sigma*X)

        return X

class RBF_Slater_NELEC_GENERAL(nn.Module):

    def __init__(self,
                input_features,
                output_features,
                centers,
                sigma,
                nelec):

        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the atoms
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF_Slater_NELEC,self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(centers)
        self.centers.requires_grad = True
        self.ncenter = len(self.centers)
        
        # get the standard deviations
        self.sigma = nn.Parameter(sigma)
        self.sigma.requires_grad = True

        # wavefunction data
        self.nelec = nelec
        self.ndim = int(self.input_features/self.nelec)

    def forward(self,input):
        
        # get the x,y,z, distance component of each point from each RBF center
        # -> (Nbatch,Nelec,Nrbf,Ndim)
        delta =  (input.view(-1,self.nelec,1,self.ndim) - self.centers[None,...])

        # compute the distance
        # -> (Nbatch,Nelec,Nrbf)
        X = (delta**2).sum(3)
        X = torch.sqrt(X)
        
        # multiply by the exponent and take the exponential
        # -> (Nbatch,Nelec,Nrbf)
        X = torch.exp(-self.sigma*X)

        return X

def SphericalHarmonics(xyz,l,m):

    '''Compute the Real Spherical Harmonics of the AO.
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, distance component of each point from each RBF center
        l : array(Nrbf) l quantum number
        m : array(Nrbf) m quantum number
    Returns:
        Y array (Nbatch,Nelec,Nrbf) : value of each SH at each point
    '''

    Y = torch.zeros(xyz.shape[:-1])

    # l=0
    ind = (l==0).nonzero().view(-1)
    Y[:,:,ind] = _spherical_harmonics_l0(xyz[:,:,ind,:])

    # l=1
    indl = (l==1)
    for mval in [-1,0,1]:
        indm = (m==mval)
        ind = (indl*indm).nonzero().view(-1)
        Y[:,:,ind] = _spherical_harmonics_l1(xyz[:,:,ind,:],mval)

    # l=2
    indl = (l==2)
    for mval in [-2,-1,0,1,2]:
        indm = (m==mval)
        ind = (indl*indm).nonzero().view(-1)
        Y[:,:,ind] = _spherical_harmonics_l2(xyz[:,:,ind,:],mval)


def _spherical_harmonics_l0(xyz):
    ''' Compute the l=0 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
    Returns
        Y00 = 1/2 \sqrt(1 / \pi)
    '''

    return 0.2820948 * torch.ones(xyz.shape[:-1])

def _spherical_harmonics_l1(xyz,m):
    ''' Compute the 1-1 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-1,0,1)
    Returns
        Y0-1 = \sqrt(3 / (4\pi)) y/r. (m=-1)
        Y00  = \sqrt(3 / (4\pi)) z/r (m=0)
        Y01  = \sqrt(3 / (4\pi)) x/r. (m=1)
    '''
    index = {-1:1,0:2,1:0}
    r = torch.sqrt((xyz**2).sum(-1))
    c = 0.4886025119029199
    return  c * xyz[:,:,:,index[m]] / r

def _spherical_harmonics_l2(xyz,m):
    ''' Compute the l=2 Spherical Harmonics
    Args:
        xyz : array (Nbatch,Nelec,Nrbf,Ndim) x,y,z, of (Point - Center)
        m : second quantum number (-2,-1,0,1,2)
    Returns
        Y2-2 = 1/2\sqrt(15/\pi) xy/r^2
        Y2-1 = 1/2\sqrt(15/\pi) yz/r^2
        Y20  = 1/4\sqrt(5/\pi) (-x^2-y^2+2z^2)/r^2
        Y21  = 1/2\sqrt(15/\pi) zx/r^2
        Y22  = 1/4\sqrt(15/\pi) (x*x-y*y)/r^2
    '''

    r2 = (xyz**2).sum(-1)

    if m == 0:
        c0 = 0.31539156525252005
        return c0 * ( -xyz[:,:,:,0]**2 - xyz[:,:,:,1]**2 + 2*xyz[:,:,:,2]**2 ) / r2
    if m == 2:
        c2 = 0.5462742152960396
        return c2 * (xyz[:,:,:,0]**2 - xyz[:,:,:,1]**2) / r2
    else :
        cm = 1.0925484305920792
        index = {-2:[0,1], -1:[1,2], 1:[2,0]}
        return cm * xyz[:,:,:,index[m][0]] * xyz[:,:,:,index[m][1]] / r2


