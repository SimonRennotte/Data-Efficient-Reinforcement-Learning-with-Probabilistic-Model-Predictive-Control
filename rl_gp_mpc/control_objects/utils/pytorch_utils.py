import torch
from math import sqrt

class Clamp(torch.autograd.Function):
	# This class allow the flow of gradient when at the boundary. Otherwise, when using the clamp function from pytorch, the action remains blocked at 0 or 1

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def normal_cdf(x, mu, sigma):
    return 0.5 * (1 + torch.erf((x - mu) / (sigma * sqrt(2))))