import torch
import numbers
from torch.nn.parameter import Parameter
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.functional import normalize
from torch.nn.utils import spectral_norm
from torch_utils import *


class myBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(myBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))

        self.reset_parameters()

    def reset_running_stats(self):
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)

        if self.training:
            # print('batch norm training')
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True)
            # print('mu, sigma shape {} {}'.format(mu.size(), sigma.size()))
            # print('current mu {}'.format(mu.transpose(0, 1)))
            # print('current sigma {}'.format(sigma.transpose(0, 1)))
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma

            # print('previous x {}'.format(x))

            x = x - mu
            x = x / (sigma + self.eps).sqrt()
            # x = x - self.running_mean
            # x = x / (self.running_var + self.eps).sqrt()
            x = x * self.weight + self.bias

            # print('current x{}'.format(x))
            # if (x != x).any():
            #     print('nan occurs in batch norm ... ')
            #     print('current x{}'.format(x))
            #     while True:
            #         sleep(1)

            x = x.view(C, N, H, W).transpose(0, 1)
            return x

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(myBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class myGroupNorm(nn.Module):
    def __init__(self, num_groups=32, num_features=0, eps=1e-5):
        super(myGroupNorm, self).__init__()
        assert num_features > 0
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class myZCANorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, n_power_iterations=10, n_eigens=32):
        super(myZCANorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.n_power_iterations = n_power_iterations
        self.n_eigens = n_eigens

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))
        self.register_buffer('running_subspace', torch.eye(num_features, num_features))

        self.reset_parameters()

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

    def forward(self, x):
        self._check_input_dim(x)

        if self.training:

            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True)
            x = x - mu
            x = x / (sigma + self.eps).sqrt()
            xxt = torch.mm(x, x.t())/(x.shape[1]-1) + torch.eye(C).cuda() * self.eps
            vlist = []
            lambdalist = []
            for i in range(self.n_eigens):
                vlist.append(torch.ones(self.num_features, 1).cuda())
            for i in range(self.n_eigens):
                v = vlist[i]
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.matmul(xxt, v), dim=0, eps=self.eps)
                    # v = self.power_layer(xxt, v)
                eig_lambda = torch.mean(torch.matmul(xxt, v)/v)
                # print('{} eig value {}'.format(i, eig_lambda))
                # sleep(1)
                if eig_lambda < 0:
                    # print('{} negative eig value is detected'.format(i))
                    break
                lambdalist.append(eig_lambda)
                xxt = xxt - torch.mm(torch.mm(xxt, v), v.t())
            xr = torch.zeros(x.t().shape).cuda()

            for i in range(len(lambdalist)):  # range(self.n_eigens):
                v = vlist[i]
                eig_lambda = lambdalist[i]
                tmp = torch.mm(torch.mm(x.t(), v), v.t())/torch.sqrt(eig_lambda+self.eps)
                xr = xr + tmp

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma

                subspace = torch.zeros(xxt.shape).cuda()
                for i in range(len(lambdalist)):  # range(self.n_eigens):
                    v = vlist[i]
                    eig_lambda = lambdalist[i]
                    subspace = subspace + torch.mm(v, v.t())/torch.sqrt(eig_lambda+self.eps)

                self.running_subspace = (1 - self.momentum) * self.running_subspace + self.momentum * subspace

            xr = xr.t() * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
            # xxt = torch.mm(x, x.t())
            # vlist = []
            # for i in range(self.n_eigens):
            #     vlist.append(torch.ones(self.num_features, 1).cuda())
            # for i in range(self.n_eigens):
            #     v = vlist[i]
            #     for _ in range(self.n_power_iterations):
            #         v = normalize(torch.matmul(xxt, v), dim=0, eps=self.eps)
            #     # eig_lambda = torch.mean(torch.matmul(xx, v)/v)
            #     xxt = xxt - torch.mm(torch.mm(xxt, v), v.t())
            #     # eig_vector = torch.mean(torch.matmul(xx, v) / v)
            # xr = torch.zeros(x.t().shape).cuda()
            # for i in range(self.n_eigens):
            #     v = vlist[i]
            #     tmp = torch.mm(torch.mm(x.t(), v), v.t())
            #     xr = xr + tmp
            #
            # xr = xr.t() * self.weight + self.bias
            # xr = xr.view(C, N, H, W).transpose(0, 1)
            x = torch.mm(x.t(), self.running_subspace).t()
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(myZCANorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class myPCANorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, n_power_iterations=10, n_eigens=32):
        super(myPCANorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.n_power_iterations = n_power_iterations
        self.n_eigens = n_eigens

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))
        self.register_buffer('running_subspace', torch.eye(num_features, num_features))

        self.reset_parameters()

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

    def forward(self, x):
        self._check_input_dim(x)

        if self.training:

            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True)
            x = x - mu
            x = x / (sigma + self.eps).sqrt()
            xxt = torch.mm(x, x.t())/(x.shape[1]-1) + torch.eye(C).cuda() * self.eps
            vlist = []
            # lambdalist = []
            for i in range(self.n_eigens):
                vlist.append(torch.ones(self.num_features, 1).cuda())
            for i in range(self.n_eigens):
                # print('debugging eigen vector list i ', vlist[i].t())
                v = vlist[i]
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.matmul(xxt, v), dim=0, eps=self.eps)
                    # v = self.power_layer(xxt, v)
                # eig_lambda = torch.mean(torch.matmul(xxt, v)/v)
                # print('{} eig value {}'.format(i, eig_lambda))
                # sleep(1)
                # if eig_lambda < 0:
                #     # print('{} negative eig value is detected'.format(i))
                #     break
                # lambdalist.append(eig_lambda)
                vlist[i] = v
                xxt = xxt - torch.mm(torch.mm(xxt, v), v.t())
                # print('candidate value is ', v.t())
                # print('updated value is ', vlist[i].t())
                # if i == 1:
                #     print('previous updated value is ', vlist[i-1].t())
                # print('sleeping for 10 seconds')
                # sleep(10)
            xr = torch.zeros(x.t().shape).cuda()

            for i in range(self.n_eigens):  # range(len(lambdalist))
                v = vlist[i]
                # eig_lambda = lambdalist[i]
                # tmp = torch.mm(torch.mm(x.t(), v), v.t())/torch.sqrt(eig_lambda+self.eps)
                tmp = torch.mm(torch.mm(x.t(), v), v.t())
                xr = xr + tmp

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma

                subspace = torch.zeros(xxt.shape).cuda()
                for i in range(self.n_eigens):  # range(len(lambdalist))
                    v = vlist[i]
                    # eig_lambda = lambdalist[i]
                    # subspace = subspace + torch.mm(v, v.t())/torch.sqrt(eig_lambda+self.eps)
                    subspace = subspace + torch.mm(v, v.t())

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

        super(myPCANorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class myPCANorm_noRec(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, n_power_iterations=10, n_eigens=64):
        super(myPCANorm_noRec, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.n_power_iterations = n_power_iterations
        self.n_eigens = n_eigens

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))
        self.register_buffer('running_subspace', torch.eye(num_features, num_features))

        self.reset_parameters()

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

    def forward(self, x):
        self._check_input_dim(x)

        if self.training:

            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True)
            x = x - mu
            x = x / (sigma + self.eps).sqrt()
            xxt = torch.mm(x, x.t())/(x.shape[1]-1) + torch.eye(C).cuda() * self.eps
            vlist = []
            # lambdalist = []
            for i in range(self.n_eigens):
                vlist.append(torch.ones(self.num_features, 1).cuda())

            print('debugging eigen vector list')
            for i in range(self.n_eigens):
                v = vlist[i]
                print('initial value is ', vlist[i].t())
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.matmul(xxt, v), dim=0, eps=self.eps)
                    # v = self.power_layer(xxt, v)
                # eig_lambda = torch.mean(torch.matmul(xxt, v)/v)
                # print('{} eig value {}'.format(i, eig_lambda))
                # sleep(1)
                # if eig_lambda < 0:
                #     # print('{} negative eig value is detected'.format(i))
                #     break
                # lambdalist.append(eig_lambda)
                xxt = xxt - torch.mm(torch.mm(xxt, v), v.t())
                print('candidate value is ', v.t())
                print('updated value is ', vlist[i].t())
                print('sleeping for 10 seconds')
                sleep(10)
            xr_list = []
            for i in range(self.n_eigens):  # range(len(lambdalist))
                v = vlist[i]
                # eig_lambda = lambdalist[i]
                # tmp = torch.mm(torch.mm(x.t(), v), v.t())/torch.sqrt(eig_lambda+self.eps)
                xr_list.append(torch.mm(x.t(), v))

            xr = torch.cat(xr_list, 1)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma

                subspace = torch.zeros(xxt.shape).cuda()
                for i in range(self.n_eigens):  # range(len(lambdalist))
                    v = vlist[i]
                    # eig_lambda = lambdalist[i]
                    # subspace = subspace + torch.mm(v, v.t())/torch.sqrt(eig_lambda+self.eps)
                    subspace = subspace + torch.mm(v, v.t())

                self.running_subspace = (1 - self.momentum) * self.running_subspace + self.momentum * subspace

            xr = xr.t() * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
            # compute eigen-vectors
            subspace = self.running_subspace.clone()
            vlist = []
            for i in range(self.n_eigens):
                vlist.append(torch.ones(self.num_features, 1).cuda())
            for i in range(self.n_eigens):
                v = vlist[i]
                for _ in range(self.n_power_iterations):
                    # v = normalize(torch.matmul(xxt, v), dim=0, eps=self.eps)
                    v = self.power_layer(subspace, v)
                subspace = subspace - torch.mm(torch.mm(subspace, v), v.t())

            xr = torch.zeros(x.t().shape).cuda()
            for i in range(self.n_eigens):  # range(len(lambdalist))
                v = vlist[i]
                tmp = torch.mm(x.t(), v)
                xr_slice = xr.narrow(1, i, 1)
                xr_slice = tmp

            # x = torch.mm(x.t(), self.running_subspace).t()
            # x = x * self.weight + self.bias
            # x = x.view(C, N, H, W).transpose(0, 1)
            xr = xr.t() * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(myPCANorm_noRec, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)