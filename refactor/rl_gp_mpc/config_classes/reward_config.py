import torch
from .utils.functions_process_config import convert_config_lists_to_tensor

class RewardConfig:
	def __init__(
		self, 
		target_state_norm=[1, 0.5, 0.5], 
		weight_state=[1, 0.1, 0.1], 
		weight_state_terminal=[10, 5, 5], 
		target_action_norm=[0.5], 
		weight_action=[0.05],
		exploration_factor=3,
		use_constraints=False, 
		state_min=[-0.1, 0.05, 0.05], 
		state_max=[1.1, 0.95, 0.925], 
		area_multiplier=1, 
	):
		self.target_state_norm = target_state_norm
		self.weight_state = weight_state
		self.weight_state_terminal = weight_state_terminal
		self.target_action_norm = target_action_norm
		self.weight_action = weight_action
		self.exploration_factor = exploration_factor

		self.use_constraints=use_constraints
		self.state_min=state_min
		self.state_max=state_max 
		self.area_multiplier=area_multiplier 
		# Computed after init
		self.target_state_action_norm = None
		self.weight_matrix_cost = None
		self.weight_matrix_cost_terminal = None

		self = convert_config_lists_to_tensor(self)
		self = combine_weight_matrix(self)
		self.target_state_action_norm = torch.cat((self.target_state_norm, self.target_action_norm))


def combine_weight_matrix(reward_config:RewardConfig):
	reward_config.weight_matrix_cost = \
		torch.block_diag(
			torch.diag(reward_config.weight_state),
			torch.diag(reward_config.weight_action))
	reward_config.weight_matrix_cost_terminal = torch.diag(reward_config.weight_state_terminal)
	return reward_config