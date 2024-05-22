# Copyright (c) 2024 Pavlo Melnyk, pavlo.melnyk@liu.se
# SPDX-License-Identifier: MIT

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def append_ones(weight) -> torch.Tensor:
    # since we learn normalized spheres, the last parameter is always 1
    # therefore, we append a constant vector of ones to the weights:
    B = weight.shape[0]

    if len(weight.shape) == 3:
        # conv1d layer
        ones_vector = torch.ones(B, 1, 1, device=weight.device)

    elif len(weight.shape) == 2:
        # linear layer
        ones_vector = torch.ones(B, 1, device=weight.device)

    else:
        raise NotImplementedError

    weight = torch.cat((weight, ones_vector), dim=1)

    return weight


def embed(x):

    if len(x.size()) == 4:
        # conv layer input
        B, M, _, N = x.size()  # batch_size x num_channels x num_points
        embed_term_1 = -torch.ones(B, M, 1, N, device=x.device)  # B x M x 1 x N
        embed_term_2 = -torch.sum(x ** 2, dim=2, keepdim=True) / 2  # along the channels D -> B x M x 1 x N

        x = torch.cat((x, embed_term_1, embed_term_2), dim=2)
        return x

    elif len(x.size()) == 3:
        # conv layer input
        B, _, N = x.size()  # batch_size x num_channels x num_points
        embed_term_1 = -torch.ones(B, 1, N, device=x.device)  # B x 1 x N
        embed_term_2 = -torch.sum(x ** 2, dim=1, keepdim=True) / 2  # along the channels D -> B x 1 x N

    elif len(x.size()) == 2:
        # linear layer input
        B, _ = x.size()  # batch_size x num_channels
        embed_term_1 = -torch.ones(B, 1, device=x.device)  # B x 1
        embed_term_2 = -torch.sum(x ** 2, dim=1, keepdim=True) / 2  # along the channels D

    else:
        raise NotImplementedError

    x = torch.cat((x, embed_term_1, embed_term_2), dim=1)
    return x



class EquivariantHyperspheresLayer(nn.Conv1d):

    def __init__(self,  in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, normalized_spheres: bool = False, num_vertices: int = 4):
        
        super().__init__(in_channels, out_channels, kernel_size)
        
        self.padding = [padding]
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.append_ones = append_ones
        self.kernel_size = kernel_size
        self.normalized_spheres = normalized_spheres
        
        self.construct_filter_banks = FilterBankConstructorND(num_vertices) 
        self.num_vertices = num_vertices
        
        self.init_weights()

        # assuming the embedded input has 2 embedding dims:
        self._init_rotations = torch.eye(in_channels - 2).repeat(out_channels, 1, 1).to(self.weight.device) # idenitties

        
    def init_weights(self):
        # initialize the weights with some random values from a uniform distribution with k = 1 / torch.sqrt(in_channels)
        k = 1 / np.sqrt(self.in_channels)
    
        if self.normalized_spheres:
            weight = torch.FloatTensor(self.out_channels, self.in_channels-1, 1).uniform_(-k, k)
        else:
            weight = torch.FloatTensor(self.out_channels, self.in_channels, 1).uniform_(-k, k)

        self.weight = torch.nn.Parameter(
            weight,
            requires_grad=True
        )

        self.bias = None
            
    @property
    def centers_radii_gamma(self):
        if self.normalized_spheres:
            weight = self.append_ones(self.weight.clone().detach())
        else:
            weight = self.weight.clone().detach()
        
        S = weight.clone().detach()
        
        gamma = S[:, -1]

        C = S[:, :-2] / gamma.unsqueeze(-1)
        r = torch.sqrt(C.norm(dim=1) ** 2 - 2 * S[:, -2] / gamma)
        return C.squeeze(-1), r.flatten(), gamma.flatten()
            

    def build_filter_bank(self) -> torch.Tensor:
        # using the tensor of learnable parameters (spheres), build spherical filter banks
        _init_rotations, _filter_bank = self.construct_filter_banks(self._weight)
       
        return _filter_bank, _init_rotations

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expected input shape is batch_size x num_vertices x num_points
        self.device = x.device
        B, D, N = x.shape

        if self.normalized_spheres:
            self._weight = self.append_ones(self.weight)
        else:
            self._weight = self.weight

        _filter_bank, _init_rotations = self.build_filter_bank()

        assert _filter_bank.shape == (self.num_vertices * self.out_channels, self.in_channels, self.kernel_size)
        assert len(x.shape) == 3 and D == self.in_channels, f'x.shape: {x.shape}, D: {D}, in_channels: {self.in_channels}'

        out = torch.conv1d(x, _filter_bank,
                           stride=self.stride,
                           padding=self.padding,
                           bias=None)

        out = out.reshape(B, self.out_channels, self.num_vertices, N)  # B x K x D x N
        out = out.permute(0, 2, 3, 1) # B x D x N x K
      
        return out, _init_rotations


class FilterBankConstructorND(nn.Module):

    def __init__(self, num_vertices=4):
        super().__init__()

        self.num_vertices = num_vertices

        # get a starting nD vetrex of the n-simplex:
        self.vertices = self.get_n_simplex_coordinates(num_vertices-1)
        
        # the change-of-basis matrix M, an element of O(n), where n = num_vertices - 1:
        self.M = torch.cat((self.vertices, self.vertices[0, 0].view(1, 1).expand(self.vertices.shape[0], 1)), dim=1)
        self.M = self.M / self.M.norm(dim=1, keepdim=True)

        ones_vec = self.vertices[0]
        self.register_buffer('ones_vec', ones_vec)

        # compute the basis rotations:
        tetra_rotations = [torch.eye(num_vertices-1).unsqueeze(0)]
        tetra_rotations += [self.compute_rotation_from_two_points(self.vertices[0:1], v.unsqueeze(0)) for v in self.vertices[1:]]
        tetra_rotations = torch.stack(tetra_rotations)
        self.register_buffer('tetra_rotations', tetra_rotations.squeeze(1))


    @staticmethod
    def get_n_simplex_coordinates(n):
        # Check if n is a positive integer
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        # see https://en.wikipedia.org/wiki/Simplex#:~:text=with%20n%20facets.-,Cartesian%20coordinates%20for%20a%20regular%20n%2Ddimensional%20simplex%20in%20Rn,-%5Bedit%5D    
        p = torch.eye(n, dtype=torch.float64) # n x n, but we want n+1 n-dimensional points
        a = (1. + math.sqrt(n+1)) / n
        ones = a * torch.ones(1, n)
        p = torch.cat((ones, p), dim=0)
        # center the coordinates:
        p = p - p.mean(dim=0, keepdim=True)
        # rescale to unit length:
        p = p / p.norm(dim=1, keepdim=True)
        
        return p


    @classmethod
    def compute_rotation_from_two_points(cls, p, q):
        ''' 
        A reflection method (thanks to https://math.stackexchange.com/a/2161631):
            assuming ||p|| == ||q||
            f(A, u) = A - 2 u (u^T S)/||u||^2
            S = f(I_n, p+q)
            R = f(S, q)
            
        Args: 
            p, q - torch.Tensor - two nD points, necessarily with ||p|| == ||q||
        
        Return:
            R - DxD rotation matrix such that R p = q
        '''    
        # check for NaN and infinite values in p and q:
        nan_indices_p = torch.nonzero(torch.isnan(p), as_tuple=True)
        nan_indices_q = torch.nonzero(torch.isnan(q), as_tuple=True)
        inf_indices_p = torch.nonzero(torch.isinf(p), as_tuple=True)
        inf_indices_q = torch.nonzero(torch.isinf(q), as_tuple=True)

        if nan_indices_p[0].numel() > 0 or nan_indices_q[0].numel() > 0:
            print(f"Input tensors p or q contain NaN values. "
                f"NaN indices in p: {nan_indices_p}, "
                f"p: {p},"
                f"q: {q},"
                f"NaN indices in q: {nan_indices_q}")

        if inf_indices_p[0].numel() > 0 or inf_indices_q[0].numel() > 0:
            print(f"Input tensors p or q contain infinite values. "
                f"Inf indices in p: {inf_indices_p}, "
                f"Inf indices in q: {inf_indices_q}")

        assert len(p.shape) == 2 and p.shape == q.shape
        a = torch.abs(p.norm(dim=1, keepdim=True).pow(2) - q.norm(dim=1, keepdim=True).pow(2)).max()

        # check for NaN and infinite values in the computation of a:
        if torch.isnan(a).any() or torch.isinf(a).any():
            print("The computation of 'a' results in NaN or infinite values.")

        assert a < 1e-5, 'Such a rotation doesn\'t exist: ||p|| must be equal to ||q||, '+str(a)
        B, D = p.shape 
        
        def reflection(S, u):   
            # reflection of S on hyperplane u:
            # S can be a matrix; S and u must have the same number of rows.
            assert len(S) == len(u) and S.shape[-1] == u.shape[-1]
            
            v = torch.matmul(u.unsqueeze(1), S) # (Bx1xD)
            v = v.squeeze(1) / u.norm(dim=1, keepdim=True)**2 # (BxD) / (Bx1) = (BxD)
            M = S - 2 * torch.matmul(u.unsqueeze(-1), v.unsqueeze(1)) # the matmul performs the outer product of u and v            
            return M
                    
        S = reflection( torch.eye(D, dtype=torch.float64).repeat(B, 1, 1).to(p.device), p+q )  # S @ p = -q, S @ q = -p
        R = reflection(S, q) # R @ p = q, R.T @ q = p            
        
        return R


    @classmethod
    def construct_rotation_isoms(cls, rotations, ndims2add=2):
        d = rotations.shape[1]
        rots = rotations.reshape(-1, d, d)
        isoms = torch.eye(d+ndims2add, device=rots.device).tile(rots.shape[0], 1, 1)
        isoms[:, :d, :d] = rots
        return isoms


    @classmethod
    def unembed_points(cls, embedded_points):
        """
        Performs a mapping that is inverse to conformal embedding.

        Args:
            embedded_points: points embedded in R^{d}, an array of shape (num_points, d).
        Returns:
            points:          nD points, an array of shape (num_points, d).
        """
        # p-normalize points, i.e., divide by the last element.
        # The first three elements are now Euclidean R^{d} coordinates:
        eps = 1e-12
        d = embedded_points.shape[-1] - 2
        points = embedded_points[:, :d] / (embedded_points[:, -1:] + eps)
        return points


    @classmethod
    def transform_points(cls, points, transformation):
        """
        Apply one or a batch of isomorphism transformations to embedded points.

        Args:
            points:              points embedded in R^{n}, an array of shape (num_points, n);
            transformation:      an array of shape (num_points, n, n) or (1, n, n) or (6, n)
        Returns:
            transformed points:  a tensor of the same shape as the input points.
        """
        n = points.shape[-1]
        T = transformation.reshape(-1, n, n)  # for performing matmul properly
        X = points.unsqueeze(-1)  # for performing matmul properly
        Y = torch.matmul(T, X)  # transform points
        Y = Y.squeeze(-1)  # to input shape

        return Y


    def forward(self, spheres, verbose=False):

        K = spheres.shape[0]
        M = self.num_vertices
        D = M - 1 # the current space dimensionality

        spheres = spheres.reshape(-1, D+2)
        # print(f"spheres: {spheres}")
        # Step 1) compute the rotations R_O^k, i.e., from the original sphere centers to the original vertex:
        centers = self.unembed_points(spheres)  # (n_spheres x D)
        # print(f"centers: {centers}")
        centers_n = centers / centers.norm(dim=1, keepdim=True)
        # print(f"centers_n: {centers_n}")
        
        ones = F.normalize(self.ones_vec.unsqueeze(0).expand(K, -1), dim=1)
        rotations_0 = self.compute_rotation_from_two_points(centers_n, ones)
        rotations_0_isom = self.construct_rotation_isoms(rotations_0)

        # rotate the original spheres into the original vertex (in R^(D+2)):
        rotated_spheres = self.transform_points(spheres, rotations_0_isom) # rotations R_O's are applied to spheres from RHS

        # Step 2) get the n-simplex rotations R_{Ti},
        # i.e. the rotations transforming the original vertex into the other M-1 vertices
        tetra_rotations = self.tetra_rotations

        # Step 3) construct the filter banks B(S):
        # rotate *directly* in the conformal R^(D+2) space
        # (already includes the INVERSE of the rotations from the original sphere centers to the original vertex):

        # Tiling so that the M-rotations are applied to all N points: (could be done with einsum)
        # tile N times as 1,2,3,4,...,N, 1,2,3,4,...,N ... :
        tetra_rotations4 = tetra_rotations.tile(K, 1, 1, 1).view(M * K, D, D)
        # tile M times as 1,1,1,...,1, 2,2,2,...,2, ... to match the tetra rotations:
        rotated_spheres4 = rotated_spheres.unsqueeze(1).tile(1, M, 1).view(M * K, D+2)
        rotations_0_4 = rotations_0.unsqueeze(1).tile(1, M, 1, 1).view(M * K, D, D)

        # actually construct the filter bank
        b = self.construct_rotation_isoms(rotations_0_4.transpose(1, 2) @ tetra_rotations4)  # R_O^T R_Ti
        filter_banks = self.transform_points(rotated_spheres4, b).unsqueeze(-1)              # R_O^T R_Ti R_O S

        if verbose:
            print('\noriginal centers:\n', centers)
            print('\nrotated original spheres[0]:', rotated_spheres[0])
            print('\noriginal spheres[0]:', spheres[0])
            print('\nrotations_0[0]:', rotations_0[0])

        return rotations_0, filter_banks