import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from torch.autograd import Variable


class FocalBinaryTverskyLoss(Function):

    def __init__(ctx, alpha=0.5, beta=0.5, gamma=1.0, reduction='mean'):
        """

        :param alpha: controls the penalty for false positives.
        :param beta: penalty for false negative.
        :param gamma : focal coefficient range[1,3]
        :param reduction: return mode
        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
        add focal index -> loss=(1-T_index)**(1/gamma)

        """
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.epsilon = 1e-6
        ctx.reduction = reduction
        ctx.gamma = gamma
        sum = ctx.beta + ctx.alpha
        if sum != 1:
            ctx.beta = ctx.beta / sum
            ctx.alpha = ctx.alpha / sum

    # @staticmethod
    def forward(ctx, input, target):
        batch_size = input.size(0)
        _, input_label = input.max(1)

        input_label = input_label.float()
        target_label = target.float()

        ctx.save_for_backward(input, target_label)

        input_label = input_label.view(batch_size, -1)
        target_label = target_label.view(batch_size, -1)

        ctx.P_G = torch.sum(input_label * target_label, 1)  # TP
        ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)  # FP
        ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)  # FN

        index = ctx.P_G / (ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.NP_G + ctx.epsilon)
        loss = torch.pow((1 - index), 1 / ctx.gamma)
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if ctx.reduction == 'none':
            loss = loss
        elif ctx.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

    # @staticmethod
    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out:
        :return:
        d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
        (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
                        = 2*P_G
        (dT_loss/d_p0)=
        """
        inputs, target = ctx.saved_tensors
        inputs = inputs.float()
        target = target.float()
        batch_size = inputs.size(0)
        sum = ctx.P_G + ctx.alpha * ctx.P_NG + ctx.beta * ctx.NP_G + ctx.epsilon
        P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
        if inputs.dim() == 5:
            sum = sum.view(batch_size, 1, 1, 1, 1)
        elif inputs.dim() == 4:
            sum = sum.view(batch_size, 1, 1, 1)
            P_G = ctx.P_G.view(batch_size, 1, 1, 1)
        sub = (ctx.alpha * (1 - target) + target) * P_G

        dL_dT = (1 / ctx.gamma) * torch.pow((P_G / sum), (1 / ctx.gamma - 1))
        dT_dp0 = -2 * (target / sum - sub / sum / sum)
        dL_dp0 = dL_dT * dT_dp0

        dT_dp1 = ctx.beta * (1 - target) * P_G / sum / sum
        dL_dp1 = dL_dT * dT_dp1
        grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
        # grad_input = torch.cat((grad_out.item() * dL_dp0, dL_dp0 * grad_out.item()), dim=1)
        return grad_input, None
