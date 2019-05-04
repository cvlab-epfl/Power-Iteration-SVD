import torch as th
# from torch.nn.functional import normalize
import torch.nn as nn
from time import sleep
'''
This file contains all the custom pytorch operator.
'''


# class power_iteration(nn.Module):
#     def __init__(self, n_power_iterations=2, eps=1e-5):
#         super(power_iteration, self).__init__()
#         self.n_power_iterations = n_power_iterations
#         self.eps = eps
#         self.power_operation = power_iteration_once.apply
#
#     def _check_input_shape(self, M, v):
#         if M.shape[0] != M.shape[1] or M.dim() != 2:
#             raise ValueError('2D covariance matrix size is {}, but it should be square'.format(M.shape()))
#         if M.shape[0] != v.shape[0]:
#             raise ValueError('input covariance dim {} should equal to eig-vector shape {})'.
#                              format(M.shape[0], v.shape[0]))
#
#     def forward(self, M, v):
#         self._check_input_shape(M, v)
#         for k in range(self.n_power_iterations):
#             v = self.power_operation(M, v)
#         return v


class power_iteration_once(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        vk_list = []
        vk_list.append(v_k)
        ctx.num_iter = num_iter
        for _ in range(int(ctx.num_iter)):
            v_k = M.mm(v_k)
            v_k /= th.norm(v_k).clamp(min=1.e-5)
            vk_list.append(v_k)

        ctx.save_for_backward(M, *vk_list)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M = ctx.saved_tensors[0]
        vk_list = ctx.saved_tensors[1:]
        dL_dvk1 = grad_output
        dL_dM = 0
        # print('dL_dvk1', dL_dvk1.t())
        # sleep(1)
        for i in range(1, ctx.num_iter + 1):
            v_k1 = vk_list[-i]
            v_k = vk_list[-i - 1]
            mid = calc_mid(M, v_k, v_k1, dL_dvk1)
            dL_dM += mid.mm(th.t(v_k))
            dL_dvk1 = M.mm(mid)
        # print('debug 1 ', dL_dM.t())
        # sleep(1)
        # if dL_dM.abs().max().item() > 0.005:
        #     import IPython
        #     IPython.embed()
        return dL_dM, dL_dvk1


def calc_mid(M, v_k, v_k1, dL_dvk1):
    M_shp = M.shape[-1]
    mid = (th.eye(M_shp).double().cuda() - v_k1.mm(th.t(v_k1)))/th.norm(M.mm(v_k)).clamp(min=1.e-5)
    # print('debug-2', mid)
    # sleep(1)
    mid = mid.mm(dL_dvk1)
    return mid