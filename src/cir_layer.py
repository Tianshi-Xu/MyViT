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
        self.alphas_after = None
        self.input = None
        search=2
        
        while search<=64 and in_features %search ==0 and out_features %search ==0:
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
        self.ILP = False
        
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
        self.alphas_after = alphas_after
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
            if self.ILP:
                # add lambda only for ILP
                assert self.weight.grad is not None
                lambda_tmp = self.weight.grad.reshape(q, block_size, p, block_size)
                lambda_tmp = lambda_tmp.permute(0, 2, 1, 3)
                lambda_tmp = lambda_tmp ** 2
                # print(lambda_tmp[0,0,:,:])
                lambda_rot = lambda_tmp[:,:, rotate_mat[:, 0], rotate_mat[:, 1]]
                lambda_rot = lambda_rot.view(q,p, block_size, block_size)
                weights_cir = lambda_rot*weights_rot

                weights_cir = torch.sum(weights_cir, dim=2, keepdim=True)/torch.sum(lambda_rot, dim=2, keepdim=True)
            else:
                weights_cir = torch.mean(weights_rot, dim=2, keepdim=True)
            # if the grad are all zero, use the average weights
            if torch.isnan(torch.mean(weights_cir)):
                weights_cir2 = torch.mean(weights_rot, dim=2, keepdim=True)
                nan_mask = torch.isnan(weights_cir)
                nan_indices = torch.nonzero(nan_mask)
                weights_cir[nan_indices] = weights_cir2[nan_indices]
            weights_cir = weights_cir.repeat(1,1, block_size, 1)
            weights_cir = weights_cir[:,:, rev_rotate_mat[:, 0], rev_rotate_mat[:, 1]] 
            weights_cir = weights_cir.view(q,p, block_size, block_size)
            # print(weights_cir[0,0,:,:])
            weights_cir=weights_cir.permute(0,2,1,3).reshape(self.out_features,self.in_features)
            weight=weight+alphas_after[idx]*weights_cir
        return weight
    
    # get the alpha after softmax, if hard, fix the block size
    def get_alphas_after(self):
        return self.gumbel_softmax()
        logits = self.alphas
        dim=-1
        if self.hard:
            # print("it is hard")
            return F.one_hot(torch.argmax(logits, dim), logits.shape[-1]).float()
        return F.softmax(logits/self.tau, dim=dim)
    
    def gumbel_softmax(self):
        logits = self.alphas
        tau = self.tau
        hard = self.hard
        dim=-1
        gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
        gumbels = gumbel_dist.sample(logits.shape)

        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
    
    def forward(self,x):
        self.input = x.detach()
        if len(x.shape)==3:
            self.d1 = x.shape[1]
        else:
            self.d1 = 1
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
    def __init__(self, in_features, out_features, kernel_size, stride,fix_block_size=-1):
        super(CirConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.feature_size = None
        self.d1 = None
        self.fix_block_size = fix_block_size
        # print("finetune:",self.finetune)
        self.padding = kernel_size//2
        self.tau = 1.0
        self.hard = False
        self.rotate_mat = {}
        self.rev_rotate_mat = {}
        self.search_space = []
        self.alphas_after = None
        self.input = None
        search=1
        while search<=64 and in_features %search ==0 and out_features %search ==0:
            self.search_space.append(search)
            search *= 2
        # self.search_space = [self.search_space[-1]]
        self.alphas = nn.Parameter(torch.ones(len(self.search_space)), requires_grad=True)

        self.weight = nn.Parameter(torch.zeros(out_features,in_features, kernel_size,kernel_size))
        # self.weight_prime = None
        self.grad = None
        self.separate_weights = None
        self.separate_weight = False
        self.alphas_after = None
        init.kaiming_uniform_(self.weight)
        self.set_rotate_mat()
        self.ILP = False
    
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
        self.alphas_after = alphas_after
        if self.separate_weight:
            if self.separate_weights is None:
                self.gen_separete_weights(device)
            weight = alphas_after[0]*self.separate_weights[0].to(device)
            for idx,block_size in enumerate(search_space):
                if idx==0:
                    continue
                rev_rotate_mat = self.rev_rotate_mat[block_size].to(device)
                q = self.out_features // block_size
                p = self.in_features // block_size
                w_i = self.separate_weights[block_size].repeat(1,1, block_size, 1,1,1).to(device)
                w_i = w_i[:,:, rev_rotate_mat[:, 0], rev_rotate_mat[:, 1],:,:] 
                w_i = w_i.view(q,p, block_size, block_size, self.kernel_size,self.kernel_size)
                # print(weights_cir[0,0,:,:])
                w_i=w_i.permute(0,2,1,3,4,5).reshape(self.out_features,self.in_features, self.kernel_size,self.kernel_size)
                weight=weight+alphas_after[idx]*w_i
        else:
            weight=torch.zeros_like(self.weight).to(device)
            for idx,block_size in enumerate(search_space):
                if block_size == 1:
                    weight = weight + alphas_after[idx]*self.weight
                    continue
                if torch.abs(alphas_after[idx]) <1e-8:
                    continue
                # print("block_size:",block_size)
                rotate_mat = self.rotate_mat[block_size].to(device)
                rev_rotate_mat = self.rev_rotate_mat[block_size].to(device)
                q = self.out_features // block_size
                p = self.in_features // block_size
                # if self.weight_prime is None:
                tmp = self.weight.reshape(q, block_size, p, block_size, self.kernel_size,self.kernel_size)
                # tmp (q,p,b,b,1,1)
                tmp = tmp.permute(0, 2, 1, 3,4,5)
                # print(tmp[0,0,:,:,0,0])
                weights_rot = tmp[:,:, rotate_mat[:, 0], rotate_mat[:, 1],:,:] 
                weights_rot = weights_rot.view(q,p, block_size, block_size, self.kernel_size,self.kernel_size)
                if self.ILP:
                    # add lambda only for ILP
                    assert self.weight.grad is not None
                    if self.grad is None:
                        self.grad = self.weight.grad.clone()
                    lambda_tmp = self.grad.reshape(q, block_size, p, block_size, self.kernel_size,self.kernel_size)
                    lambda_tmp = lambda_tmp.permute(0, 2, 1, 3,4,5)
                    lambda_tmp = lambda_tmp * 1e5
                    lambda_tmp = lambda_tmp ** 2
                    
                    # print(lambda_tmp[0,0,:,:,0,0])
                    lambda_rot = lambda_tmp[:,:, rotate_mat[:, 0], rotate_mat[:, 1],:,:]
                    lambda_rot = lambda_rot.view(q,p, block_size, block_size, self.kernel_size,self.kernel_size)
                    weights_cir = lambda_rot*weights_rot
                    weights_cir = torch.sum(weights_cir, dim=2, keepdim=True)/torch.sum(lambda_rot, dim=2, keepdim=True)
                else:
                    weights_cir = torch.mean(weights_rot, dim=2, keepdim=True)
                # if the grad are all zero, use the average weights
                if torch.isnan(torch.mean(weights_cir)):
                    weights_cir2 = torch.mean(weights_rot, dim=2, keepdim=True)
                    nan_mask = torch.isnan(weights_cir)
                    nan_indices = torch.nonzero(nan_mask)
                    weights_cir[nan_indices] = weights_cir2[nan_indices]
                weights_cir = weights_cir.repeat(1,1, block_size, 1,1,1)
                weights_cir = weights_cir[:,:, rev_rotate_mat[:, 0], rev_rotate_mat[:, 1],:,:] 
                weights_cir = weights_cir.view(q,p, block_size, block_size, self.kernel_size,self.kernel_size)
                # print(weights_cir[0,0,:,:,0,0])
                weights_cir = weights_cir.permute(0,2,1,3,4,5).reshape(self.out_features,self.in_features, self.kernel_size,self.kernel_size)
                weight=weight+alphas_after[idx]*weights_cir
        return weight
    
    def gen_separete_weights(self,device):
        self.separate_weights = {}
        self.separate_weights[0] = self.weight
        search_space = self.search_space
        for idx,block_size in enumerate(search_space):
            if idx==0:
                continue
            q = self.out_features // block_size
            p = self.in_features // block_size
            self.separate_weights[block_size] = nn.Parameter(torch.zeros(q,p,1,block_size,self.kernel_size,self.kernel_size),requires_grad=True)
            # print("block_size:",block_size)
            rotate_mat = self.rotate_mat[block_size].to(device)
            rev_rotate_mat = self.rev_rotate_mat[block_size].to(device)
            tmp = self.weight.reshape(q, block_size, p, block_size, self.kernel_size,self.kernel_size)
            # tmp (q,p,b,b,1,1)
            tmp = tmp.permute(0, 2, 1, 3,4,5)
            # print(tmp[0,0,:,:])
            weights_rot = tmp[:,:, rotate_mat[:, 0], rotate_mat[:, 1],:,:] 
            # (q,p,b,b,1,1)
            weights_rot = weights_rot.view(q,p, block_size, block_size, self.kernel_size,self.kernel_size)
            # print("-----------")
            # (q,p,b,1,1)
            weights_cir = torch.mean(weights_rot, dim=2, keepdim=True)
            with torch.no_grad():
                self.separate_weights[block_size].copy_(weights_cir.data)
            
        
    def gumbel_softmax(self):
        logits = self.alphas
        tau = self.tau
        hard = self.hard
        dim=-1
        gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(0.0001, device=logits.device, dtype=logits.dtype))
        gumbels = gumbel_dist.sample(logits.shape)

        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
    
    def get_alphas_after(self):
        return self.gumbel_softmax()
        logits = self.alphas
        dim=-1
        if self.hard:
            # print("it is hard")
            return F.one_hot(torch.argmax(logits, dim), logits.shape[-1]).float()
        return F.softmax(logits/self.tau, dim=dim)
    
    def forward(self, x):
        if self.feature_size is None:
            assert x.size(2) == x.size(3)
            self.feature_size = x.size(2)
            self.d1 = self.feature_size ** 2
        self.input = x.detach()
        weight=self.trans_to_cir_meng(x.device)
        x = F.conv2d(x,weight,None,self.stride,self.padding)
        return x

    def get_final_block_size(self):
        return self.search_space[torch.argmax(self.alphas)]
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, kernel_size={self.kernel_size}, fix_block_size={self.fix_block_size}, search_space={self.search_space}, separate_weight={self.separate_weight}'

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
        