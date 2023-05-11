import torch

class ObservationConfig:
	def __init__(
		self,
		obs_var_norm:"list[float]"=[1e-6, 1e-6, 1e-6]
	):
		self.obs_var_norm = torch.diag(torch.Tensor(obs_var_norm))

