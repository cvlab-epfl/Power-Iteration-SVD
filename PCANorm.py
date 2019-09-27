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

from tensorboardX import SummaryWriter
import os


n_eigens_debug = 20
# save_dir = 'runs'
# logdir = os.path.join(save_dir, 'ResNet18_adv_adapt', '{}-bs{}'.format('pcanorm', 128),
#                       'pi-eig{}-trail2'.format(n_eigens_debug))
# writer = SummaryWriter(log_dir=logdir)
# global step
# step = 0


# def save_grad_(grad):
#     global step
#     writer.add_scalar('grad/{}mean'.format('dL_dM_'), grad.abs().mean().item(), step)
#     writer.add_scalar('grad/{}max'.format('dL_dM_'), grad.abs().max().item(), step)
#     step += 1


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
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, n_power_iterations=19):
        super(myZCANorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.n_power_iterations = n_power_iterations
        self.n_eigens = int(num_features/2)

        self.weight = Parameter(torch.Tensor(num_features, 1).double())
        self.bias = Parameter(torch.Tensor(num_features, 1).double())
        # self.power_layer = power_iteration()
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1).double())
        self.register_buffer('running_var', torch.ones(num_features, 1).double())
        self.register_buffer('running_subspace', torch.eye(num_features, num_features).double())

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
            self.register_buffer("{}".format(i), th.ones(self.num_features, 1, requires_grad=True).double())
        self.eig_dict = self.state_dict()

    def reset_dictionary(self):
        for i in range(self.n_eigens):
            self.eig_dict[str(i)].data.normal_(0, 1)
            self.eig_dict[str(i)].data /= torch.norm(self.eig_dict[str(i)].data)
            self.state_dict()[str(i)].copy_(self.eig_dict[str(i)])
            # self.eig_dict[str(i)] = torch.ones(self.num_features, 1, requires_grad=True).double()
            # self.eig_dict[str(i)].grad.zero_()

    def forward(self, x):
        self._check_input_dim(x)
        # self.reset_dictionary()
        self.eig_dict = self.state_dict()

        if self.training:
            x = x.double()
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True)

            x = x - mu
            x = x / (sigma + self.eps).sqrt()
            xxt = torch.mm(x, x.t())/(N*H*W) + torch.eye(C).cuda().double() * self.eps

            lambdalist = []
            # lambda_sum = 0
            lambda_sum_gt = 0
            with torch.no_grad():

                u, e, v = torch.svd(xxt)
                for i in range(self.n_eigens):
                    self.eig_dict[str(i)] = v[:, i][..., None]

            for i in range(self.n_eigens):
                # print('{}-th eig-vector initial norm & value is {} \n {}'.
                #       format(i, self.eig_dict[str(i)].norm().item(), self.eig_dict[str(i)].t()))
                self.eig_dict[str(i)] = self.power_layer(xxt, self.eig_dict[str(i)])
                # xxt.register_hook(save_grad_)

                eig_lambda = torch.matmul(self.eig_dict[str(i)].t(), torch.matmul(xxt, self.eig_dict[str(i)]))
                lambdalist.append(eig_lambda)
                # lambda_sum += eig_lambda
                # energy_lower_bound = lambda_sum/(lambda_sum + eig_lambda*(C-(i+1)))

                lambda_sum_gt += e[i]
                energy_lower_bound_gt = lambda_sum_gt/e.sum()  # (lambda_sum_gt + eig_lambda*(C-(i+1)))
                # print('{}-Mnorm:{} eig-value {} at least {} energy '.
                #       format(i+1, xxt.norm().item(), eig_lambda.item(), energy_lower_bound.item()))
                # sleep(0.5)

                # if energy_lower_bound >= 0.95:
                if energy_lower_bound_gt >= 0.95:
                    break
                xxt = xxt - torch.mm(torch.mm(xxt, self.eig_dict[str(i)]), self.eig_dict[str(i)].t())
            xr = torch.zeros(x.t().shape).double().cuda()

            for i in range(len(lambdalist)):  # range(self.n_eigens)
                # v = vlist[i]
                # eig_lambda = lambdalist[i]
                tmp = torch.mm(torch.mm(x.t(), self.eig_dict[str(i)]/lambdalist[i].sqrt()), self.eig_dict[str(i)].t())
                xr = xr + tmp

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma

                subspace = torch.zeros(xxt.shape).double().cuda()
                # for i in range(self.n_eigens):
                for i in range(len(lambdalist)):
                    v = self.eig_dict[str(i)]
                    eig_lambda = lambdalist[i]
                    subspace = subspace + torch.mm(v, v.t())/torch.sqrt(eig_lambda+self.eps)
                    # subspace = subspace + torch.mm(v, v.t())

                self.running_subspace = (1 - self.momentum) * self.running_subspace + self.momentum * subspace

            xr = xr.t() * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr.float()

        else:
            N, C, H, W = x.size()
            x = x.double()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
            x = torch.mm(x.t(), self.running_subspace).t()
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x.float()

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(myZCANorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ZCANormPI(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormPI, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0
                # print(xxtj[i])
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0)/e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            print('{}/{} eigen-vectors selected'.format(j + 1, length))
                            print(e[0:counter_i])
                            break
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        eigenvector_ij.data = v[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    # eigenvector_ij.register_hook(print)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if lambda_ij < 0:
                        print('eigenvalues: ', e)
                        # sys.exit("Error message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                    if diff_ratio > 0.1:
                        # print('inaccurate eigenvalue computed: ', e)
                        # sys.exit("Error message: inaccurate PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        # print("Warning message: inaccurate PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    # print('lambda_group{}_{}: {} vs gt {}'.format(i, j, lambda_ij.item(), e[j].item()))
                    # if j>0:
                    #     print('SVD eig-{} (angle) between current and dominant eigenvectors'.format(j),
                    #           F.cosine_similarity(v[:, j][..., None], v[:, 0][..., None], dim=0).clamp(-1.0,1.0).abs().acos().item() / math.pi * 180)
                    # if lambda_ij<0:
                    #     import IPython
                    #     IPython.embed()
                    # remove projections on the eigenvectors
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)



def print_grad(grad):
    print('hi')
    print(grad)
    sleep(5)


class myPCANorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, n_power_iterations=19):
        super(myPCANorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.n_power_iterations = n_power_iterations
        self.n_eigens = int(num_features/4)

        self.weight = Parameter(torch.Tensor(num_features, 1).double())
        self.bias = Parameter(torch.Tensor(num_features, 1).double())
        # self.power_layer = power_iteration()
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1).double())
        self.register_buffer('running_var', torch.ones(num_features, 1).double())
        self.register_buffer('running_subspace', torch.eye(num_features, num_features).double())

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
            self.register_buffer("{}".format(i), th.ones(self.num_features, 1, requires_grad=True).double())

        self.eig_dict = self.state_dict()

    def reset_dictionary(self):
        for i in range(self.n_eigens):
            self.eig_dict[str(i)].data.normal_(0, 1)
            self.eig_dict[str(i)].data /= torch.norm(self.eig_dict[str(i)].data)
            self.state_dict()[str(i)].copy_(self.eig_dict[str(i)])
            # self.eig_dict[str(i)] = torch.ones(self.num_features, 1, requires_grad=True).double()
            # self.eig_dict[str(i)].grad.zero_()

    def forward(self, x):
        self._check_input_dim(x)
        # self.reset_dictionary()
        self.eig_dict = self.state_dict()

        if self.training:
            x = x.double()
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            sigma = x.var(1, keepdim=True)

            x = x - mu
            x = x / (sigma + self.eps).sqrt()
            xxt = torch.mm(x, x.t())/(N*H*W) + torch.eye(C).cuda().double() * self.eps

            lambdalist = []
            # lambda_sum = 0
            lambda_sum_gt = 0
            with torch.no_grad():
                # e, v = torch.symeig(xxt, eigenvectors=True)
                # e = e.flip([0])
                # v = v.flip([1])
                time_start = time.time()
                u, e, v = torch.svd(xxt)
                for i in range(self.n_eigens):
                    self.eig_dict[str(i)] = v[:, i][..., None]
                time_end = time.time()
                print('svd computation time {:0.6f} ms'.format((time_end - time_start) * 1000))

            time_start = time.time()
            for i in range(self.n_eigens):
                # print('{}-th eig-vector initial norm & value is {} \n {}'.
                #       format(i, self.eig_dict[str(i)].norm().item(), self.eig_dict[str(i)].t()))
                self.eig_dict[str(i)] = self.power_layer(xxt, self.eig_dict[str(i)])
                # xxt.register_hook(save_grad_)

                eig_lambda = torch.matmul(self.eig_dict[str(i)].t(), torch.matmul(xxt, self.eig_dict[str(i)]))
                lambdalist.append(eig_lambda)
                # lambda_sum += eig_lambda
                # energy_lower_bound = lambda_sum/(lambda_sum + eig_lambda*(C-(i+1)))

                lambda_sum_gt += e[i]
                energy_lower_bound_gt = lambda_sum_gt/e.sum()  # (lambda_sum_gt + eig_lambda*(C-(i+1)))
                # print('{}-Mnorm:{} eig-value {} at least {} energy '.
                #       format(i+1, xxt.norm().item(), eig_lambda.item(), energy_lower_bound.item()))
                # sleep(0.5)

                # if energy_lower_bound >= 0.95:
                if energy_lower_bound_gt >= 0.95:
                    break
                xxt = xxt - torch.mm(torch.mm(xxt, self.eig_dict[str(i)]), self.eig_dict[str(i)].t())
            time_end = time.time()
            print('power layer forward time {:0.6f} ms\n'.format((time_end - time_start) * 1000))
            xr = torch.zeros(x.t().shape).double().cuda()

            for i in range(len(lambdalist)):  # range(self.n_eigens)
                # v = vlist[i]
                # eig_lambda = lambdalist[i]
                tmp = torch.mm(torch.mm(x.t(), self.eig_dict[str(i)]), self.eig_dict[str(i)].t())
                xr = xr + tmp

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma

                subspace = torch.zeros(xxt.shape).double().cuda()
                # for i in range(self.n_eigens):
                for i in range(len(lambdalist)):
                    v = self.eig_dict[str(i)]
                    # eig_lambda = lambdalist[i]
                    # subspace = subspace + torch.mm(v, v.t())/torch.sqrt(eig_lambda+self.eps)
                    subspace = subspace + torch.mm(v, v.t())

                self.running_subspace = (1 - self.momentum) * self.running_subspace + self.momentum * subspace

            xr = xr.t() * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr.float()

        else:
            N, C, H, W = x.size()
            x = x.double()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
            x = torch.mm(x.t(), self.running_subspace).t()
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x.float()

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(myPCANorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class myPCANormfloat(nn.Module):
    def __init__(self, num_features, eps=1e-4, momentum=0.1, affine=True, n_power_iterations=19):
        super(myPCANormfloat, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.n_power_iterations = n_power_iterations
        self.n_eigens = 2  # num_features  # int(num_features/2)

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
            # xxt = torch.mm(x, x.t())/(N*H*W) + torch.eye(C, out=torch.empty_like(x)) * self.eps
            xxt = torch.mm(x, x.t())/(N*H*W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            counter = 0
            lambda_sum_gt = 0
            with torch.no_grad():
                u, e, v = torch.svd(xxt)
                # ei = e[0:-1]
                # eip1 = e[1::]
                # ei_eip1 = (eip1/ei).mean()
                # print(ei_eip1.item())
                for i in range(self.n_eigens):
                    self.eig_dict[str(i)] = v[:, i][..., None]

            for i in range(self.n_eigens):
                self.eig_dict[str(i)] = self.power_layer(xxt, self.eig_dict[str(i)])
                counter += 1
                xxt = xxt - torch.mm(torch.mm(xxt, self.eig_dict[str(i)]), self.eig_dict[str(i)].t())
                # if e[i].item() < self.eps:
                #     print('eigenvalue smaller than threshold {}..'.format(self.eps))
                #     break
                # self.eig_dict[str(i)] = self.power_layer(xxt, self.eig_dict[str(i)])
                # counter += 1
                # lambda_sum_gt += e[i]
                # xxt = xxt - torch.mm(torch.mm(xxt, self.eig_dict[str(i)]), self.eig_dict[str(i)].t())
                # energy_lower_bound_gt = lambda_sum_gt / e.sum()
                # if energy_lower_bound_gt >= 0.99:
                #     # print('{}/{} eigen-vectors selected'.format(i+1, self.num_features))
                #     break

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

        super(myPCANormfloat, self)._load_from_state_dict(
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
        self.power_layer = power_iteration_once.apply
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
