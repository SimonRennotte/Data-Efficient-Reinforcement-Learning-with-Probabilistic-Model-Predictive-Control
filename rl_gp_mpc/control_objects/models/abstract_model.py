import torch

from rl_gp_mpc.config_classes.model_config import ModelConfig

class AbstractStateTransitionModel:
	def __init__(self, config:ModelConfig, dim_state, dim_action):
		self.config = config
		self.dim_state = dim_state
		self.dim_action = dim_action
		self.dim_input = self.dim_state + self.dim_action

	def predict_trajectory(self, input:torch.Tensor, input_var:torch.Tensor) -> "tuple[torch.Tensor]":
		raise NotImplementedError

	def predict_next_state(self, input: torch.Tensor, input_var: torch.Tensor) -> "tuple[torch.Tensor]":
		raise NotImplementedError

	def prepare_inference(self, x, y):
		raise NotImplementedError

	def train(self, x, y):
		raise NotImplementedError

	def save_state(self):
		raise NotImplementedError

	def load_state(self, saved_state):
		raise NotImplementedError
