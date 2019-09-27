import torch as th
# from torch.nn.functional import normalize
import torch.nn as nn
from time import sleep
import time
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


class power_iteration_unstable(th.autograd.Function):
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
        # print('befor. mat-bp gradients {}'.format(dL_dvk1))
        # print('befor. mat-bp gradients {0:0.6f}'.format(dL_dvk1.abs().max().item()))
        for i in range(1, ctx.num_iter + 1):
            v_k1 = vk_list[-i]
            v_k = vk_list[-i - 1]
            mid = calc_mid(M, v_k, v_k1, dL_dvk1)
            dL_dM += mid.mm(th.t(v_k))
            dL_dvk1 = M.mm(mid)
        # print('after. mat-bp gradients {0:0.6f}'.format(dL_dM.abs().max().item()))
        return dL_dM, dL_dvk1


def calc_mid(M, v_k, v_k1, dL_dvk1):
    I = th.eye(M.shape[-1], out=th.empty_like(M))
    mid = (I - v_k1.mm(th.t(v_k1)))/th.norm(M.mm(v_k)).clamp(min=1.e-5)
    mid = mid.mm(dL_dvk1)
    return mid


class power_iteration_once(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = th.eye(M.shape[-1], out=th.empty_like(M))
        numerator = I - v_k.mm(th.t(v_k))
        denominator = th.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = th.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak


class svd_future(th.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        """
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        """
        u, s, vh = th.svd(M)
        ctx.save_for_backward(M, u, s, vh)
        return u, s, vh

    @staticmethod
    def backward(ctx, dL_du, dL_ds, dL_dv):
        M, u, s, vh, e_gt = ctx.saved_tensors

        s_diag = th.diag(s)
        dL_ds = th.diag(dL_ds)
        # proj = vh.t().mm(e_gt[..., None])
        # dL_ds = th.diag(proj.flatten()) * 10.
        v = vh.t()
        uh = u.t()
        s_2 = s**2

        # K_t = 1.0 / (s_2[None] - s_2[..., None] + I) - I
        # K = K_t.t()

        K_t = geometric_approximation(s_2).t()  # using geometric series to approximate the K

        # manually set the gradient of the eigen-value
        manu_s = - th.ones((9,))
        manu_s[-1] = 1.
        dL_ds = th.diag(manu_s)

        D = th.matmul(dL_du, th.diag(1.0 / s))
        Dh = D.t()

        inner = K_t * (vh.mm(dL_dv - v.mm(Dh).mm(u).mm(s_diag)))
        dL_dM = D.mm(v) + u.mm(dL_ds - uh.mm(D)).mm(vh) + \
                2 * u.mm(s_diag).mm(sym(inner)).mm(vh)

        return dL_dM, th.zeros_like(e_gt)


def geometric_approximation(s):
    I = th.eye(s.shape[0])
    p = s[..., None] / s[None] - I
    p = th.where(p < 1., p, 1. / p)
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    a1 = 1. / th.where(a1 >= a1_t, a1, - a1_t)
    a1 *= th.ones(s.shape[0], s.shape[0]) - I
    p_hat = th.ones_like(p)
    for i in range(19):
        p_hat = p_hat * p
        a1 += a1 * p_hat
    return a1


class svdv2(th.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        s, u = th.symeig(M, eigenvectors=True, upper=True)  # s in a ascending sequence.
        ctx.save_for_backward(M, u, s)
        return u, s

    @staticmethod
    def backward(ctx, dL_du, dL_ds):
        M, u, s = ctx.saved_tensors
        # I = th.eye(s.shape[0])
        # K_t = 1.0 / (s[None] - s[..., None] + I) - I
        K_t = geometric_approximation(s).t()
        u_t = u.t()
        dL_dM = u.mm(K_t * u_t.mm(dL_du) + th.diag(dL_ds)).mm(u_t)
        return dL_dM

