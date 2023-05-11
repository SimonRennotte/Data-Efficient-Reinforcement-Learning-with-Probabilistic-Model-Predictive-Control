import torch

class Clamp(torch.autograd.Function):
	# This class allow the flow of gradient when at the boundary. Otherwise, when using the clamp function from pytorch, the action remains blocked at 0 or 1

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None