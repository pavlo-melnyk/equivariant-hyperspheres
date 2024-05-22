# Copyright (c) 2024 Pavlo Melnyk, pavlo.melnyk@liu.se
# SPDX-License-Identifier: MIT

import torch 
import numpy as np 

from .hyperspheres import embed, EquivariantHyperspheresLayer

from torch import nn
import torch.nn.functional as F
from engineer.metrics.metrics import MetricCollection, Loss, Accuracy

    
class NormalizationLayer(nn.Module):
    def __init__(self, features, init: float = 0, nonperm=False, requires_grad=True):
        super().__init__()
        self.in_features = features # e.g., number of spheres in a layer
  
        if nonperm:
            self.forward = self.forward_nonperm
        
        self.a = nn.Parameter(torch.zeros(self.in_features) + init, requires_grad=requires_grad)

    def forward_nonperm(self, x):
       # x is assumed to be of shape B x D x N x K
        B, D, N, K = x.shape
        assert self.in_features == N*K

        norms = x.norm(dim=1, keepdim=True)                                          # B x 1 x N x K  
        s_a = torch.sigmoid(self.a).view(1, 1, N, K).repeat(B, 1, 1, 1).to(x.device) # B x 1 x N x K
        scaled_norms = s_a * (norms - 1) + 1                                         # interpolates between 1 and the norm
        normalized = x / (scaled_norms + 1e-12)                                      # B x D x N x K

        return normalized
    
    def forward(self, x, dim=1):
        # x is assumed to be of shape B x D x N x K
        B, D, N, K = x.shape
        assert K == self.in_features

        norms = x.norm(dim=dim, keepdim=True)                                        # B x 1 x N x K  
        max_norms = norms.amax(dim=2, keepdim=True)                                  # B x 1 x 1 x K
        s_a = torch.sigmoid(self.a).view(1, 1, 1, K).repeat(B, 1, 1, 1).to(x.device) # B x 1 x 1 x K
        max_norms = s_a * (max_norms - 1) + 1                                        # interpolates between 1 and the norm
        normalized = x / (max_norms + 1e-12)

        return normalized


class DEH_hulls(nn.Module):
    def __init__(self, depth: int = 1, width: list = [1], hidden_features: int = 32, output_channels: int = 1, space_dim: int = 5, num_points: int = 16,  nonlinear_init: float = 0.0,
                 sphere_bias: bool = False, normalized_spheres: bool = False, fix_spheres: bool = False, norm_activations_grad: bool = True, permutation_invariant: bool = True, debug: bool = False): 
        super().__init__()

        self.depth = depth
        self.width = width
        self.permutation_invariant = permutation_invariant
        self.embed = embed

        assert len(width) == self.depth
        K_prod = np.prod(self.width) if len(self.width) > 0 else 1

        if not debug:
            self.train_metrics = MetricCollection({"loss": Loss(),},)
            self.test_metrics = MetricCollection({"loss": Loss(),})
        else:
            self.forward = self.debug_forward

        self.num_points = num_points
        self.normalized_spheres = normalized_spheres
        self.leaky_relu = True
        self.space_dim = space_dim
        self.embed = embed 

        edims = 2        
        
        self.spheres = nn.ModuleList()
        self.biases = nn.ParameterList()
        self.norm_layers = nn.ModuleList()
        p = 1 if permutation_invariant else num_points
        self.p = p

        for i in range(self.depth):
            steerable_layer = EquivariantHyperspheresLayer(space_dim + edims, np.prod(self.width[:i+1]), 1, num_vertices=space_dim + 1, normalized_spheres=normalized_spheres)
            bias = torch.nn.Parameter(torch.zeros(1, p*np.prod(self.width[:i+1])), requires_grad=sphere_bias)
            norm_layer = NormalizationLayer(p*np.prod(self.width[:i+1]), nonperm=not permutation_invariant, requires_grad=norm_activations_grad, init=nonlinear_init)

            if fix_spheres:
                for par in steerable_layer.parameters():
                    par.requires_grad = False

            self.spheres.append(steerable_layer)
            self.biases.append(bias)
            self.norm_layers.append(norm_layer)

            space_dim += 1
           
        # self.M = self.spheres[0].construct_filter_banks.M
        # print(self.M)

        m = 1 if permutation_invariant else self.num_points//2
        
        self.linear1    = nn.Linear(2*K_prod*self.num_points*m, hidden_features)
        self.linear2    = nn.Linear(hidden_features, output_channels)

    def debug_forward(self, batch):
        x = batch
        x = x.to(torch.float)

        x = x.view(len(x), self.num_points, self.space_dim)

        x = x.permute(0, 2, 1)

        B, D, N = x.shape  # B x D x N
        K = 1

        for i in range(self.depth):                        
            x = x.reshape(B, D, -1)
            x, _ = self.spheres[i](self.embed(x))       # B x D x N*K x self.width[i]   (or  B x D x N x self.width  if i==0)
            D += 1

            x = x.reshape(B, D, N, K, K, self.width[i]) # np.prod(self.width[:i]) = K * self.width[i]
            x = torch.diagonal(x, dim1=3, dim2=4)       # B x D x N x self.width[i] x K
            x = x.transpose(-1, -2)                     # B x D x N x K x self.width[i]


            K = K * self.width[i]
            
            x = x.reshape(B, D, N, -1)
            x = x + self.biases[i].view(1, 1, self.p, K).expand(B, D, N, K).to(x.device)            # B x D x N x K               

            x = self.norm_layers[i](x)                  # B x D x N x K
     
            x = x.reshape(B, D, -1)    # B x D x N*K

        x = x.reshape(B, D, N, K)
        x = torch.einsum("bndk,bdmk->bnmk", x.transpose(2, 1), x)

        if self.permutation_invariant:
            # make permutation-invariant:
            x, idcs = torch.sort(x, dim=-2) # the other N dimension is order dependent because of the autoproduct y
            x1 = torch.amax(x, dim=1, keepdim=True) # B x 1 x N x K
            x2 = torch.mean(x, dim=1, keepdim=True) # B x 1 x N x K
            x = torch.cat([x1, x2], dim=1)          # B x 2 x N x K
        
        x = x.reshape(B, -1)                    # B x 2*N*K
        x = F.silu(self.linear1(x))

        y = self.linear2(x)
        return y
   
    
    def forward(self, batch, step):
        x, yl = batch
        yl = yl.unsqueeze(-1)
        x = x.to(torch.float)

        x = x.view(len(x), self.num_points, self.space_dim)

        x = x.permute(0, 2, 1)        

        B, D, N = x.shape  # B x D x N
        K = 1

        for i in range(self.depth):                        
            x = x.reshape(B, D, -1)
            x, _ = self.spheres[i](self.embed(x))       # B x D x N*K x self.width[i]   (or  B x D x N x self.width  if i==0)

            D += 1

            x = x.reshape(B, D, N, K, K, self.width[i]) # np.prod(self.width[:i]) = K * self.width[i]
            x = torch.diagonal(x, dim1=3, dim2=4)       # B x D x N x self.width[i] x K
            x = x.transpose(-1, -2)                     # B x D x N x K x self.width[i]

            K = K * self.width[i]
            
            x = x.reshape(B, D, N, -1)
            x = x + self.biases[i].view(1, 1, self.p, K).expand(B, D, N, K).to(x.device)            # B x D x N x K               
           
            # else:
            x = self.norm_layers[i](x) # B x D x N x K
            x = x.reshape(B, D, -1)    # B x D x N*K

        # compute autoproduct:
        x = x.reshape(B, D, N, K)
        x = torch.einsum("bndk,bdmk->bnmk", x.transpose(2, 1), x)

        if self.permutation_invariant:
            # make permutation-invariant:
            x, idcs = torch.sort(x, dim=-2) # the other N dimension is order-dependent because of the autoproduct y
            x1 = torch.amax(x, dim=1, keepdim=True) # B x 1 x N x K
            x2 = torch.mean(x, dim=1, keepdim=True) # B x 1 x N x K
            x = torch.cat([x1, x2], dim=1)          # B x 2 x N x K
        
        x = x.reshape(B, -1)                        # B x 2*N*K
        x = F.silu(self.linear1(x))

        y = self.linear2(x)
        
        assert y.shape == yl.shape
        loss = F.mse_loss(y, yl.float(), reduction="none")
        return loss.mean(), {"loss": loss,}


class DEH_o5reg(nn.Module):
       
    def __init__(self, ymean, ystd, depth: int = 0, width: list = [1], hidden_features: int = 32, output_channels: int = 1, space_dim: int = 5, num_points: int = 2,  nonlinear_init: float = 0.0,
                 sphere_bias: bool = False, normalized_spheres: bool = False, fix_spheres: bool = False, norm_activations_grad: bool = True, permutation_invariant: bool = False, debug: bool = False): 
        super().__init__()

        self.ymean = ymean
        self.ystd = ystd
        
        self.depth = depth
        self.width = width
        self.permutation_invariant = permutation_invariant
        self.embed = embed

        assert len(width) == self.depth
        K_prod = np.prod(self.width) if len(width) > 0 else 1

        if not debug:
            self.train_metrics = MetricCollection({"loss": Loss(),},)
            self.test_metrics = MetricCollection({"loss": Loss(),})
        else:
            self.forward = self.debug_forward

        self.num_points = num_points
        self.normalized_spheres = normalized_spheres
        self.leaky_relu = True
        self.space_dim = space_dim
       
        edims = 2        
        
        self.spheres = nn.ModuleList()
        self.biases = nn.ParameterList()
        self.norm_layers = nn.ModuleList()
        p = 1 if permutation_invariant else num_points
        self.p = p

        for i in range(self.depth):
            steerable_layer = EquivariantHyperspheresLayer(space_dim + edims, np.prod(self.width[:i+1]), 1, num_vertices=space_dim + 1, normalized_spheres=normalized_spheres)
            bias = torch.nn.Parameter(torch.zeros(1, p*np.prod(self.width[:i+1])), requires_grad=sphere_bias)
            norm_layer = NormalizationLayer(p*np.prod(self.width[:i+1]), nonperm=not permutation_invariant, requires_grad=norm_activations_grad, init=nonlinear_init)

            if fix_spheres:
                for p in steerable_layer.parameters():
                    p.requires_grad = False

            self.spheres.append(steerable_layer)
            self.biases.append(bias)
            self.norm_layers.append(norm_layer)

            space_dim += 1
           
        # self.M = self.spheres[0].construct_filter_banks.M
        # print(self.M)

        self.linear1    = nn.Linear(K_prod*(self.num_points**2 + self.num_points)//2, hidden_features)
        self.bn1        = nn.Identity()
        self.linear2    = nn.Linear(hidden_features, output_channels)

    def debug_forward(self, batch):
        x = batch
        x = x.to(torch.float)

        x = x.view(len(x), self.num_points, self.space_dim)
        x = x.permute(0, 2, 1)        

        B, D, N = x.shape  # B x D x N
        K = 1

        for i in range(self.depth):                        
            x = x.reshape(B, D, -1)
            x, _ = self.spheres[i](self.embed(x))       # B x D x N*K x self.width[i]   (or  B x D x N x self.width  if i==0)
            # if not self.keep_nd:
            D += 1

            x = x.reshape(B, D, N, K, K, self.width[i]) # np.prod(self.width[:i]) = K * self.width[i]
            x = torch.diagonal(x, dim1=3, dim2=4)       # B x D x N x self.width[i] x K
            x = x.transpose(-1, -2)                     # B x D x N x K x self.width[i]
            K = K * self.width[i]
            
            x = x.reshape(B, D, N, -1)
            x = x + self.biases[i].view(1, 1, self.p, K).expand(B, D, N, K).to(x.device) # B x D x N x K               

            x = self.norm_layers[i](x) # B x D x N x K
     
            x = x.reshape(B, D, -1)    # B x D x N*K


        # compute autoproduct:
        x = x.reshape(B, D, N, K)
        x = torch.einsum("bndk,bdmk->bnmk", x.transpose(2, 1), x)

        x[:, 0, -1, :] = 2**0.5 * x[:, 0, -1, :]
        triu_ab = torch.triu_indices(N, N)  # 2 x num_N, where num_N = (N^2 + N)/2
        x = x[:, triu_ab[0], triu_ab[1], :] # B x num_N x K
        
        x = x.reshape(B, -1)                # B x 2*N*K or B x num_N*K
        x = F.silu(self.linear1(x))

        y = self.linear2(x)

        return y
   
    
    def forward(self, batch, step):
        x, yl = batch
        x = x.to(torch.float)

        x = x.view(len(x), self.num_points, self.space_dim)
        x = x.permute(0,2,1)        

        B, D, N = x.shape  # B x D x N
        K = 1

        for i in range(self.depth):                        
            x = x.reshape(B, D, -1)
            x, _ = self.spheres[i](self.embed(x))       # B x D x N*K x self.width[i]   (or  B x D x N x self.width  if i==0)

            D += 1

            x = x.reshape(B, D, N, K, K, self.width[i]) # np.prod(self.width[:i]) = K * self.width[i]
            x = torch.diagonal(x, dim1=3, dim2=4)       # B x D x N x self.width[i] x K
            x = x.transpose(-1, -2)                     # B x D x N x K x self.width[i]

            K = K * self.width[i]
            
            x = x.reshape(B, D, N, -1)
            x = x + self.biases[i].view(1, 1, self.p, K).expand(B, D, N, K).to(x.device)    # B x D x N x K               
           
            # else:
            x = self.norm_layers[i](x) # B x D x N x K
     
            x = x.reshape(B, D, -1)    # B x D x N*K

        # compute autoproduct:
        x = x.reshape(B, D, N, K)
        x = torch.einsum("bndk,bdmk->bnmk", x.transpose(2, 1), x)

        x[:, 0, -1, :] = 2**0.5 * x[:, 0, -1, :]
        triu_ab = torch.triu_indices(N, N)  # 2 x num_N, where num_N = (N^2 + N)/2
        x = x[:, triu_ab[0], triu_ab[1], :] # B x num_N x K
        
        x = x.reshape(B, -1)                # B x 2*N*K or B x num_N*K
        x = F.silu(self.linear1(x))

        y = self.linear2(x)

        normalized_y = y * self.ystd + self.ymean
        normalized_yl = yl * self.ystd + self.ymean
        
        assert y.shape == yl.shape, breakpoint()
        loss = F.mse_loss(normalized_y, normalized_yl.float(), reduction="none")
        return loss.mean(), {"loss": loss,}
    

class DEH_skeletons(nn.Module):
    
    def __init__(self, depth: int = 1, width: list = [1], hidden_features: int = 32, output_channels: int = 10, space_dim: int = 3, num_points: int = 20,  nonlinear_init: float = 0.0,
                 sphere_bias: bool = False, normalized_spheres: bool = False, fix_spheres: bool = False, norm_activations_grad: bool = True, permutation_invariant: bool = True, debug: bool = False): 
        super().__init__()

        self.depth = depth
        self.width = width
        self.permutation_invariant = permutation_invariant
        self.embed = embed
       
        assert len(width) == self.depth
        K_prod = np.prod(self.width)
   
        if not debug:        
            self.train_metrics = MetricCollection(dict(loss=Loss(), acc=Accuracy()))
            self.test_metrics = MetricCollection(dict(loss=Loss(), acc=Accuracy()))
        else:
            self.forward = self.debug_forward

        self.num_points = num_points
        self.normalized_spheres = normalized_spheres
        self.leaky_relu = True
        self.space_dim = space_dim
       
        edims = 2        
        
        self.spheres = nn.ModuleList()
        self.biases = nn.ParameterList()
        self.norm_layers = nn.ModuleList()
        p = 1 if permutation_invariant else num_points
        self.p = p

        for i in range(self.depth):
            steerable_layer = EquivariantHyperspheresLayer(space_dim + edims, np.prod(self.width[:i+1]), 1, num_vertices=space_dim + 1, normalized_spheres=normalized_spheres)
            bias = torch.nn.Parameter(torch.zeros(1, p*np.prod(self.width[:i+1])), requires_grad=sphere_bias)
            norm_layer = NormalizationLayer(p*np.prod(self.width[:i+1]), nonperm=not permutation_invariant, requires_grad=norm_activations_grad, init=nonlinear_init)

            if fix_spheres:
                for p in steerable_layer.parameters():
                    p.requires_grad = False

            self.spheres.append(steerable_layer)
            self.biases.append(bias)
            self.norm_layers.append(norm_layer)

            space_dim += 1

        self.M = self.spheres[0].construct_filter_banks.M
        # print(self.M)

        m = 1 if permutation_invariant else self.num_points//2

        self.linear1    = nn.Linear(2*K_prod*self.num_points*m, hidden_features)
        self.bn1        = nn.Identity()
        self.linear2    = nn.Linear(hidden_features, output_channels)

    def debug_forward(self, batch):
        x = batch
        x = x.permute(0,2,1)
        B, D, N = x.shape  # B x D x N

        K = 1

        for i in range(self.depth):                        
            x = x.reshape(B, D, -1)
            
            x, _ = self.spheres[i](self.embed(x))       # B x D x N*K x self.width[i]   (or  B x D x N x self.width  if i==0)

            D += 1

            x = x.reshape(B, D, N, K, K, self.width[i]) # np.prod(self.width[:i]) = K * self.width[i]
            x = torch.diagonal(x, dim1=3, dim2=4)       # B x D x N x self.width[i] x K
            x = x.transpose(-1, -2)                     # B x D x N x K x self.width[i]

            K = K * self.width[i]
            
            x = x.reshape(B, D, N, -1)
            x = x + self.biases[i].view(1, 1, self.p, K).expand(B, D, N, K).to(x.device)    # B x D x N x K               

            x = self.norm_layers[i](x) # B x D x N x K
     
            x = x.reshape(B, D, -1)    # B x D x N*K

        # compute autoproduct:
        x = x.reshape(B, D, N, K)
        x = torch.einsum("bndk,bdmk->bnmk", x.transpose(2, 1), x)

        if self.permutation_invariant:
            # make permutation-invariant:
            x, idcs = torch.sort(x, dim=-2) # the other N dimension is order dependent because of the autoproduct y
            x1 = torch.amax(x, dim=1, keepdim=True) # B x 1 x N x K
            x2 = torch.mean(x, dim=1, keepdim=True) # B x 1 x N x K
            x = torch.cat([x1, x2], dim=1)          # B x 2 x N x K
        
        x = x.reshape(B, -1)                        # B x 2*N*K
        x = F.silu(self.linear1(x))

        y = self.linear2(x)
        return y
   
    def forward(self, batch, step):
        x, labels = batch
        x = x.permute(0, 2, 1)
        B, D, N = x.shape  # B x D x N     
        K = 1

        for i in range(self.depth):                        
            x = x.reshape(B, D, -1)
            x, _ = self.spheres[i](self.embed(x))       # B x D x N*K x self.width[i]   (or  B x D x N x self.width  if i==0)

            D += 1

            x = x.reshape(B, D, N, K, K, self.width[i]) # np.prod(self.width[:i]) = K * self.width[i]
            x = torch.diagonal(x, dim1=3, dim2=4)       # B x D x N x self.width[i] x K
            x = x.transpose(-1, -2)                     # B x D x N x K x self.width[i]


            K = K * self.width[i]
            
            x = x.reshape(B, D, N, -1)
            x = x + self.biases[i].view(1, 1, self.p, K).expand(B, D, N, K).to(x.device)    # B x D x N x K               

            x = self.norm_layers[i](x) # B x D x N x K
     
            x = x.reshape(B, D, -1)    # B x D x N*K

        # compute autoproduct:
        x = x.reshape(B, D, N, K)
        x = torch.einsum("bndk,bdmk->bnmk", x.transpose(2, 1), x)

        if self.permutation_invariant:
            # make permutation-invariant:
            x, idcs = torch.sort(x, dim=-2) # the other N dimension is order dependent because of the autoproduct y
            x1 = torch.amax(x, dim=1, keepdim=True) # B x 1 x N x K
            x2 = torch.mean(x, dim=1, keepdim=True) # B x 1 x N x K
            x = torch.cat([x1, x2], dim=1)          # B x 2 x N x K
        
        x = x.reshape(B, -1)                        # B x 2*N*K
        x = F.silu(self.linear1(x))

        y = self.linear2(x)
        
        loss = F.cross_entropy(y, labels, reduction="none")
        acc = torch.argmax(y.detach(), dim=1) == labels

        return loss.mean(), dict(loss=loss, acc=acc)