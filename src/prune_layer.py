import math,time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class PruneConv2d(nn.Module):
    # (feature_size*feature_size,in_features) * (in_features,out_features)-->(m,n)*(n,k)
    # feature_size*feature_size*block_size<=4096
    def __init__(self, in_features, out_features, kernel_size, stride,prune_ratio=0):
        super(PruneConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size//2
        self.prune_ratio = prune_ratio
        self.weight = nn.Parameter(torch.zeros(out_features,in_features, kernel_size,kernel_size))
        self.mask = torch.ones_like(self.weight.data)
        self.alphas_after = None
        init.kaiming_uniform_(self.weight)
        self.prune_weight()
    
    def prune_weight(self):
        K = self.weight.size(0)
        C = self.weight.size(1)
        current_ratio = 0
        if K>=C:
            diag_idx=C-1
            while current_ratio<self.prune_ratio: 
                for i in range(int(math.floor(K/C))):
                    diag = torch.diagonal(self.mask[i*C:(i+1)*C,:,:,:],diag_idx,0,1)
                    diag.zero_()
                    diag2 = torch.diagonal(self.mask[i*C:(i+1)*C,:,:,:],-C+diag_idx,0,1)
                    diag2.zero_()
                diag_idx-=1
                current_ratio = 1-torch.sum(self.mask).item()/torch.numel(self.mask)
            print(self)
            print("current_ratio",current_ratio)
        else:
            diag_idx=K-1
            while current_ratio<self.prune_ratio: 
                for i in range(int(math.floor(C/K))):
                    diag = torch.diagonal(self.mask[:,i*K:(i+1)*K,:,:],diag_idx,0,1)
                    diag.zero_()
                    diag2 = torch.diagonal(self.mask[:,i*K:(i+1)*K,:,:],-K+diag_idx,0,1)
                    diag2.zero_()
                diag_idx-=1
                current_ratio = 1-torch.sum(self.mask).item()/torch.numel(self.mask)
            print(self)
            print("current_ratio",current_ratio)
        self.prune_ratio = current_ratio
    
    def forward(self, x):
        weight=self.weight*self.mask.to(x.device)
        x = F.conv2d(x,weight,None,self.stride,self.padding)
        return x
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, kernel_size={self.kernel_size}, prune_ratio={self.prune_ratio}'


if __name__ == '__main__':
    conv = PruneConv2d(16, 96, 1, 1,0.5)
    x = torch.randn(1,16,32,32)
    y = conv(x)