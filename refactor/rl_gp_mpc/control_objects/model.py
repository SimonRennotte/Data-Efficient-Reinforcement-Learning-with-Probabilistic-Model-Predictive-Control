import torch


class AbstractStateTransitionModel:
	def __init__(self, config_model):
		self.config = config_model

	def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def prepare_inference(self):
		raise NotImplementedError

	def predict_trajectory(self, state_0:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
		raise NotImplementedError


class StateTransitionModel(AbstractStateTransitionModel):
	def __init__(self, config_model):
		super().__init__(config_model)

	def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		pass

	def prepare_inference(self):
		pass

	def predict_trajectory(self, state_0: torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
		pass
