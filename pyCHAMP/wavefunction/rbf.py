import torch
from torch import nn

class RBF(nn.Module):

    def __init__(self,input_features,output_features,centers):
        super(RBF,self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.centers = nn.Parameter(torch.Tensor(centers))
        self.sigma = (centers[1]-centers[0])

        self.weight = nn.Parameter(torch.Tensor(output_features,input_features))
        self.weight.data.uniform_(-0.1,0.1)
        self.weight.data.fill_(1.)
        self.weight.requires_grad = False

        #self.bias = nn.Parameter(torch.Tensor(self.centers.shape))
        self.register_parameter('bias',None)

    def forward(self,input):
        out = nn.functional.linear(input,self.weight,self.bias)
        return torch.exp( -(out-self.centers)**2 / self.sigma ) 