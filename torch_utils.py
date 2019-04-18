import torch as th
from torch.nn.functional import normalize

'''
This file contains all the custom pytorch operator.
'''


class power_iteration(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        v_k1 = M.mm(v_k)
        v_k1 /= normalize(v_k1, dim=0, eps=1e-5)
        # v_k1 /= th.norm(v_k1)
        ctx.save_for_backward(M, v_k, v_k1)
        return v_k1

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k, v_k1 = ctx.saved_tensors
        dL_dvk1 = grad_output
        M_shp = M.shape[-1]
        # mid = (th.eye(M_shp).to("cuda") - v_k1.mm(th.t(v_k1)))/th.norm(M.mm(v_k))
        mid = (th.eye(M_shp).cuda() - v_k1.mm(th.t(v_k1)))/th.norm(M.mm(v_k))
        mid = mid.mm(dL_dvk1)
        dL_dM = mid.mm(th.t(v_k))
        dL_dvk = M.mm(mid)
        return dL_dM, dL_dvk