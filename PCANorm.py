import torch
import numbers
from torch.nn.parameter import Parameter
from time import sleep
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.functional import normalize
from torch.nn.utils import spectral_norm
from torch_utils import *


class myPCANormSVDPI(nn.Module):
    def __init__(self, num_features, eps=1e-4, momentum=0.1, affine=True, n_power_iterations=19):
        super(myPCANormSVDPI, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.n_power_iterations = n_power_iterations
        self.n_eigens = 16  # num_features  # int(num_features/2)

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))
        self.register_buffer('running_subspace', torch.eye(num_features, num_features))

        self.reset_parameters()
        self.create_dictionary()

    def reset_running_stats(self):
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.running_subspace.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def create_dictionary(self):
        for i in range(self.n_eigens):
            self.register_buffer("{}".format(i), th.ones(self.num_features, 1, requires_grad=True))

        self.eig_dict = self.state_dict()

    def forward(self, x):
        self._check_input_dim(x)
        self.eig_dict = self.state_dict()

        if self.training:
            x = x
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True)

            x = x - mu
            x = x / (sigma + self.eps).sqrt()
            xxt = torch.mm(x, x.t())/(N*H*W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            counter = 0
            lambda_sum_gt = 0
            with torch.no_grad():
                u, e, v = torch.svd(xxt)
                for i in range(self.n_eigens):
                    self.eig_dict[str(i)] = v[:, i][..., None]

            for i in range(self.n_eigens):
                self.eig_dict[str(i)] = self.power_layer(xxt, self.eig_dict[str(i)])
                counter += 1
                xxt = xxt - torch.mm(torch.mm(xxt, self.eig_dict[str(i)]), self.eig_dict[str(i)].t())

            xr = torch.zeros_like(x.t())
            for i in range(counter):
                xr += torch.mm(torch.mm(x.t(), self.eig_dict[str(i)]), self.eig_dict[str(i)].t())

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma
                subspace = torch.zeros_like(xxt)
                for i in range(counter):
                    subspace = subspace + torch.mm(self.eig_dict[str(i)], self.eig_dict[str(i)].t())
                self.running_subspace = (1 - self.momentum) * self.running_subspace + self.momentum * subspace

            xr = xr.t() * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
            x = torch.mm(x.t(), self.running_subspace).t()
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(myPCANormSVDPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
