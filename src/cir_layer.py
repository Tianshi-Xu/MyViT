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
        # fix_block_size represents uniform block size
        self.fix_block_size = fix_block_size
        # hard is used for finetune
        self.hard = False
        self.tau = 1.0
        # d1 = input.shape[0] used for compute the latency
        self.d1 = None
        # search_space = [2,4,8,16] or [16], block size=1 always exists
        self.rotate_mat = {}
        self.rev_rotate_mat = {}
        self.search_space = [1]
        search=2
        
        while search<=16 and in_features %search ==0 and out_features %search ==0:
            self.search_space.append(search)
            search *= 2
        
        # weight for each block size
        self.alphas = nn.Parameter(torch.ones(len(self.search_space)), requires_grad=True)
        self.weight = nn.Parameter(torch.zeros(out_features,in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()
        self.set_rotate_mat()
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def set_rotate_mat(self):
        for block_size in self.search_space:
            if block_size==1:
                continue
            rotate_mat = torch.zeros(block_size * block_size, 2).int()
            rev_rotate_mat = torch.zeros(block_size * block_size, 2).int()
            for i in range(0, block_size):
                for j in range(0, block_size):
                    rotate_mat[i * block_size + j, 0] = i 
                    rotate_mat[i * block_size + j, 1] = (i + j) % block_size

                    rev_rotate_mat[i * block_size + j, 0] = i 
                    rev_rotate_mat[i * block_size + j, 1] = (j - i) % block_size
            self.rotate_mat[block_size] = rotate_mat
            self.rev_rotate_mat[block_size] = rev_rotate_mat
            
    # weight = \sum alpha[i]*W[i], alpha need to be softmaxed
    def trans_to_cir(self,device):
        search_space = self.search_space
        # if fix_block_size, directly use the block size
        if self.fix_block_size!=-1:
            if search_space[-1] < self.fix_block_size:
                alphas_after=torch.tensor([1 if i==int(math.log2(search_space[-1])) else 0 for i in range(self.alphas.shape[-1])]).to(device)
            else:
                alphas_after=torch.tensor([1 if 2**i==self.fix_block_size else 0 for i in range(self.alphas.shape[-1])]).to(device)
        else:
            alphas_after = self.get_alphas_after()
        weight=(alphas_after[0]*self.weight).to(device)
        for idx,block_size in enumerate(search_space):
            if idx==0:
                continue
            if torch.abs(alphas_after[idx]) <1e-6:
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
                tmp_compress[:,:,i] = mean_of_diagonal

            for i in range(block_size):
                w[:,:,:,i] = tmp_compress.roll(shifts=i,dims=2)
            # print(w[0,0,:,:])
            w = w.permute(0,2,1,3).reshape(self.out_features,self.in_features)
            weight=weight+alphas_after[idx]*w
        return weight
    
    def trans_to_cir_meng(self,device):
        search_space = self.search_space
        # if fix_block_size, directly use the block size
        if self.fix_block_size!=-1:
            if search_space[-1] < self.fix_block_size:
                alphas_after=torch.tensor([1 if i==int(math.log2(search_space[-1])) else 0 for i in range(self.alphas.shape[-1])]).to(device)
            else:
                alphas_after=torch.tensor([1 if 2**i==self.fix_block_size else 0 for i in range(self.alphas.shape[-1])]).to(device)
        else:
            alphas_after = self.get_alphas_after()
        weight=(alphas_after[0]*self.weight).to(device)
        for idx,block_size in enumerate(search_space):
            if idx==0:
                continue
            if torch.abs(alphas_after[idx]) <1e-6:
                continue
            # print("block_size:",block_size)
            rotate_mat = self.rotate_mat[block_size].to(device)
            rev_rotate_mat = self.rev_rotate_mat[block_size].to(device)
            q = self.out_features // block_size
            p = self.in_features // block_size
            tmp = self.weight.reshape(q, block_size, p, block_size)
            # tmp (q,p,b,b)
            tmp = tmp.permute(0, 2, 1, 3)
            # print(tmp[0,0,:,:])
            weights_rot = tmp[:,:, rotate_mat[:, 0], rotate_mat[:, 1]] 
            weights_rot = weights_rot.view(q,p, block_size, block_size)
            # print("-----------")
            weights_cir = torch.mean(weights_rot, dim=2, keepdim=True)
            weights_cir = weights_cir.repeat(1,1, block_size, 1)
            weights_cir = weights_cir[:,:, rev_rotate_mat[:, 0], rev_rotate_mat[:, 1]] 
            weights_cir = weights_cir.view(q,p, block_size, block_size)
            # print(weights_cir[0,0,:,:])
            weights_cir=weights_cir.permute(0,2,1,3).reshape(self.out_features,self.in_features)
            weight=weight+alphas_after[idx]*weights_cir
        return weight
    
    # get the alpha after softmax, if hard, fix the block size
    def get_alphas_after(self):
        logits = self.alphas
        dim=-1
        if self.hard:
            # print("it is hard")
            return F.one_hot(torch.argmax(logits, dim), logits.shape[-1]).float()
        return F.softmax(logits/self.tau, dim=dim)
    
    def get_final_block_size(self):
        return self.search_space[torch.argmax(self.alphas)]
    
    def forward(self,x):
        # print("x.shape:",x.shape)
        
        # print("w.shape:",self.weight.shape)
        if len(x.shape)==3:
            self.d1 = x.shape[1]
        else:
            self.d1 = 1
        # print("d1:",self.d1)
        weight = self.trans_to_cir_meng(x.device).to(x.device)
        y = F.linear(x, weight, self.bias)
        # print("y.shape:",y.shape)
        return y
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, fix_block_size={self.fix_block_size}, search_space={self.search_space}'

# bias is Flase by default!
class CirConv2d(nn.Module):
    # (feature_size*feature_size,in_features) * (in_features,out_features)-->(m,n)*(n,k)
    # feature_size*feature_size*block_size<=4096
    def __init__(self, in_features, out_features, kernel_size, stride,feature_size,fix_block_size=-1):
        super(CirConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.feature_size = feature_size
        self.d1 = feature_size * feature_size
        self.fix_block_size = fix_block_size
        # print("finetune:",self.finetune)
        self.padding = kernel_size//2
        self.tau = 1.0
        self.hard = False
        self.rotate_mat = {}
        self.rev_rotate_mat = {}
        self.search_space = [1,]
        search=2
        while search<=16 and in_features %search ==0 and out_features %search ==0:
            self.search_space.append(search)
            search *= 2
        # self.search_space = [self.search_space[-1]]
        self.alphas = nn.Parameter(torch.ones(len(self.search_space)), requires_grad=True)

        self.weight = nn.Parameter(torch.zeros(out_features,in_features, kernel_size,kernel_size))
        self.alphas_after = None
        init.kaiming_uniform_(self.weight)
        self.set_rotate_mat()
    
    def set_rotate_mat(self):
        for block_size in self.search_space:
            if block_size==1:
                continue
            rotate_mat = torch.zeros(block_size * block_size, 2).int()
            rev_rotate_mat = torch.zeros(block_size * block_size, 2).int()
            for i in range(0, block_size):
                for j in range(0, block_size):
                    rotate_mat[i * block_size + j, 0] = i 
                    rotate_mat[i * block_size + j, 1] = (i + j) % block_size

                    rev_rotate_mat[i * block_size + j, 0] = i 
                    rev_rotate_mat[i * block_size + j, 1] = (j - i) % block_size
            self.rotate_mat[block_size] = rotate_mat
            self.rev_rotate_mat[block_size] = rev_rotate_mat
    
    def trans_to_cir_meng(self,device):
        search_space = self.search_space
        # if fix_block_size, directly use the block size
        if self.fix_block_size!=-1:
            if search_space[-1] < self.fix_block_size:
                alphas_after=torch.tensor([1 if i==int(math.log2(search_space[-1])) else 0 for i in range(self.alphas.shape[-1])]).to(device)
            else:
                alphas_after=torch.tensor([1 if 2**i==self.fix_block_size else 0 for i in range(self.alphas.shape[-1])]).to(device)
        else:
            alphas_after = self.get_alphas_after()
        weight=(alphas_after[0]*self.weight).to(device)
        for idx,block_size in enumerate(search_space):
            if idx==0:
                continue
            if torch.abs(alphas_after[idx]) <1e-6:
                continue
            # print("block_size:",block_size)
            rotate_mat = self.rotate_mat[block_size].to(device)
            rev_rotate_mat = self.rev_rotate_mat[block_size].to(device)
            q = self.out_features // block_size
            p = self.in_features // block_size
            tmp = self.weight.reshape(q, block_size, p, block_size, self.kernel_size,self.kernel_size)
            # tmp (q,p,b,b,1,1)
            tmp = tmp.permute(0, 2, 1, 3,4,5)
            # print(tmp[0,0,:,:])
            weights_rot = tmp[:,:, rotate_mat[:, 0], rotate_mat[:, 1],:,:] 
            weights_rot = weights_rot.view(q,p, block_size, block_size, self.kernel_size,self.kernel_size)
            # print("-----------")
            weights_cir = torch.mean(weights_rot, dim=2, keepdim=True)
            weights_cir = weights_cir.repeat(1,1, block_size, 1,1,1)
            weights_cir = weights_cir[:,:, rev_rotate_mat[:, 0], rev_rotate_mat[:, 1],:,:] 
            weights_cir = weights_cir.view(q,p, block_size, block_size, self.kernel_size,self.kernel_size)
            # print(weights_cir[0,0,:,:])
            weights_cir=weights_cir.permute(0,2,1,3,4,5).reshape(self.out_features,self.in_features, self.kernel_size,self.kernel_size)
            weight=weight+alphas_after[idx]*weights_cir
        return weight
    
    def trans_to_cir(self,device):
        search_space = self.search_space
        if self.fix_block_size!=-1:
            if search_space[-1] < self.fix_block_size:
                alphas_after=torch.tensor([1 if i==int(math.log2(search_space[-1])) else 0 for i in range(self.alphas.shape[-1])]).to(device)
            else:
                alphas_after=torch.tensor([1 if 2**i==self.fix_block_size else 0 for i in range(self.alphas.shape[-1])]).to(device)
        else:
            alphas_after = self.get_alphas_after()
        # weight=torch.zeros(self.out_features,self.in_feat*ures, self.kernel_size,self.kernel_size).cuda()
        weight=alphas_after[0]*self.weight
        for idx,block_size in enumerate(search_space):
            if idx==0:
                continue
            if torch.abs(alphas_after[idx]) <1e-6:
                continue
            q=self.out_features//block_size
            p=self.in_features//block_size
            tmp = self.weight.view(q,block_size, p, block_size, self.kernel_size,self.kernel_size)
            tmp = tmp.permute(0,2,1,3,4,5)
            w = torch.zeros(q,p,block_size,block_size,self.kernel_size,self.kernel_size).to(device)
            # print(tmp[0,0,:,:,0,0])
            tmp_compress = torch.zeros(q,p,block_size,self.kernel_size,self.kernel_size).to(device)
            for i in range(block_size):
                diagonal = torch.diagonal(tmp, offset=i, dim1=2, dim2=3)
                if i>0:
                    diagonal2 = torch.diagonal(tmp, offset=-block_size+i, dim1=2, dim2=3)
                    diagonal = torch.cat((diagonal,diagonal2),dim=4)
                assert diagonal.shape[4] == block_size
                mean_of_diagonal = diagonal.mean(dim=4)
                tmp_compress[:,:,i,:,:] = mean_of_diagonal
            for i in range(block_size):
                w[:,:,:,i,:,:] = tmp_compress.roll(shifts=i, dims=2)
            # print(w[0,0,:,:,0,0])
            w = w.permute(0,2,1,3,4,5)
            w = w.reshape(q*block_size,p*block_size,self.kernel_size,self.kernel_size)
            weight=weight+alphas_after[idx]*w
        return weight
    
    def get_alphas_after(self):
        logits = self.alphas
        dim=-1
        if self.hard:
            # print("it is hard")
            return F.one_hot(torch.argmax(logits, dim), logits.shape[-1]).float()
        return F.softmax(logits/self.tau, dim=dim)
    
    def forward(self, x):
        weight=self.trans_to_cir_meng(x.device)
        x = F.conv2d(x,weight,None,self.stride,self.padding)
        return x

    def get_final_block_size(self):
        return self.search_space[torch.argmax(self.alphas)]
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, kernel_size={self.kernel_size}, fix_block_size={self.fix_block_size}, search_space={self.search_space}'

class CirBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,block_size=-1):
        super(CirBatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.block_size = block_size
    
    def forward(self, input):
        self._check_input_dim(input)
        # print("numfeatures:",self.num_features)
        # print("block_size:",self.block_size)
        if self.block_size==-1:
            tmp = self.weight
        else:
            tmp = self.weight.reshape(self.num_features//self.block_size,self.block_size)
            tmp = tmp.mean(dim=1,keepdim=True)
            tmp = tmp.repeat(1,self.block_size)
            tmp = tmp.reshape(-1)
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            tmp,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}, block_size={self.block_size}'



if __name__ == '__main__':
    # trans_to_cir_meng()
    # conv = CirLinear(16,96,-1,False)
    # x = torch.randn(4,16)
    # y = conv(x)
    x = torch.tensor([0.6543, 0.0022, 0.0044, 0.0204, 0.3187])
    y = F.softmax(x/0.20,dim=-1)
    print(y)
    # print(y)
    # K=4
    # C=4
    # H=4
    # W=4
    # h=3
    # conv = nn.Conv2d(C,K,h,padding=0,bias=False)
    # weight = torch.zeros_like(conv.weight)
    # bias=0
    # for i in range(0,C):
    #     kernel = torch.tensor([bias+j+1 for j in range(0,h*h)])
    #     bias += h*h
    #     weight[0,i,:,:] = kernel.reshape(h,h)
    # for i in range(1,K):
    #     weight[i,:,:,:] = weight[0].roll(shifts=i,dims=0)
    # print(weight[0])
    # print("----------")
    # print(weight[1])
    # bias = 1
    # x = torch.zeros((C,H,W))
    # for i in range(C):
    #     xi = torch.tensor([0,0,0,0,0,bias,bias+1,0,0,bias+2,bias+3,0,0,0,0,0]).reshape(H,W)
    #     bias += 4
    #     x[i,:,:] = xi
    # print(x)
    # conv.weight.data = weight
    # y = conv(x)
    # print(y)
    
    # x_hat = []
    # for i in range(C):
    #     xi_hat = [0 for i in range(H*W)]
    #     xi_hat[5]=x[i,1,1].item()
    #     xi_hat[6]=x[i,1,2].item()
    #     xi_hat[9]=x[i,2,1].item()
    #     xi_hat[10]=x[i,2,2].item()
    #     x_hat = x_hat + xi_hat
    # w_hat = []
    # O=W*(h-1)+h-1
    # for i in range(C):
    #     wi_hat = [0 for i in range(H*W)]
    #     for j in range(h):
    #         for k in range(h):
    #             wi_hat[O-W*j-k]=weight[i,0,j,k].item()
    #     w_hat = w_hat + wi_hat
    # # print("x_hat:",x_hat)
    # # print("w_hat:",w_hat)
    # y_hat = [0 for i in range(64)]
    # # y_hat=w_hat*x_hat
    # mod=64
    # for i in range(len(x_hat)):
    #     for j in range(len(w_hat)):
    #         y_hat[(i+j)%mod] += x_hat[i]*w_hat[j]
    # # print("y_hat:",y_hat)
    # result = torch.zeros_like(y)
    # for i in range(K):
    #     for j in range(2):
    #         for k in range(2):
    #             result[i,j,k] = y_hat[i*16+O+j*W+k]
    # print("result:",result)
        