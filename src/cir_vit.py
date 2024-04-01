import math,time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class CirLinear(nn.Module):
    def __init__(self,in_features,out_features,fix_block_size=-1,bias: bool = True):
        super(CirLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fix_block_size = fix_block_size
        self.hard = False
        self.tau = 1.0
        self.d1 = None
        self.search_space = []
        search=2
        while search<=16 and in_features %search ==0 and out_features %search ==0:
            self.search_space.append(search)
            search *= 2
        self.alphas = nn.Parameter(torch.ones(len(self.search_space)+1), requires_grad=True)
        self.weight = nn.Parameter(torch.zeros(out_features,in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
    # fix_block_size表示uniform block size, hard表示nas后固定路径
    def trans_to_cir(self,device):
        search_space = self.search_space
        if self.fix_block_size!=-1:
            # print(self.fix_block_size)
            if 2**len(search_space) < self.fix_block_size:
                alphas_after=torch.tensor([1 if i==len(search_space) else 0 for i in range(self.alphas.shape[-1])]).to(self.weight.device)
            else:
                alphas_after=torch.tensor([1 if 2**i==self.fix_block_size else 0 for i in range(self.alphas.shape[-1])]).to(self.weight.device)
            # print(alphas_after)
        else:
            alphas_after = self.get_alpha_after()
        weight=(alphas_after[0]*self.weight).to(device)
        for idx,block_size in enumerate(search_space):
            if torch.abs(alphas_after[idx+1]) <1e-6:
                continue
            q = self.out_features // block_size
            p = self.in_features // block_size
            assert self.out_features % block_size == 0
            assert self.in_features % block_size == 0
            tmp = self.weight.reshape(q, block_size, p, block_size)
            tmp = tmp.permute(0, 2, 1, 3)
            # print(tmp[0,0,:,:])
            w = torch.zeros(q, p, block_size, block_size).to(device)
            tmp_compress = torch.zeros(q,p,block_size).to(device)
            for i in range(block_size):
                diagonal = torch.diagonal(tmp,offset=i,dim1=2,dim2=3)
                if i>0:
                    part_two = torch.diagonal(tmp,offset=-block_size+i,dim1=2,dim2=3)
                    diagonal = torch.cat([diagonal,part_two],dim=2)
                mean_of_diagonal = diagonal.mean(dim=2)
                # mean_of_diagonal.shape (q,p)
                tmp_compress[:,:,i] = mean_of_diagonal

            for i in range(block_size):
                w[:,:,:,i] = tmp_compress.roll(shifts=i,dims=2)
            # print(w[0,0,:,:])
            w = w.permute(0,2,1,3).reshape(self.out_features,self.in_features)
            weight=weight+alphas_after[idx+1]*w
        return weight
    
    def get_alpha_after(self):
        logits = self.alphas
        dim=-1
        if self.hard:
            # print("it is hard")
            return F.one_hot(torch.argmax(logits, dim), logits.shape[-1]).float()
        return F.softmax(logits/self.tau, dim=dim)
    
    def forward(self,x):
        self.d1 = x.shape[0]
        weight = self.trans_to_cir(x.device).to(x.device)
        return F.linear(x, weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, fix_block_size={self.fix_block_size}, search_space={self.search_space}'
    
if __name__ == '__main__':
    cir = CirLinear(16,32,fix_block_size=4)
    x = torch.randn(2,16)
    print(cir)
    out = cir(x)