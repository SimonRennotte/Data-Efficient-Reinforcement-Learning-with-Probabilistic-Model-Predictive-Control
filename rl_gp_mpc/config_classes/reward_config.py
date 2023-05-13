import torch
from .utils.functions_process_config import convert_config_lists_to_tensor

class RewardConfig:
	def __init__(
		self, 
		target_state_norm:"list[float]"=[1, 0.5, 0.5], 
		weight_state:"list[float]"=[1, 0.1, 0.1], 
		weight_state_terminal:"list[float]"=[10, 5, 5], 
		target_action_norm:"list[float]"=[0.5], 
		weight_action:"list[float]"=[0.05],
		exploration_factor:float=3,
		use_constraints:bool=False, 
		state_min:"list[float]"=[-0.1, 0.05, 0.05], 
		state_max:"list[float]"=[1.1, 0.95, 0.925], 
		area_multiplier:float=1, 
		clip_lower_bound_cost_to_0:bool=False
	):
		"""
		target_state_norm: target states. The cost relative to observations will be 0 if the normalized observations match this. Dim=No
		weight_state: target states. The weight attributed to each observation. Dim=No
		weight_state_terminal. The weight attributed to each observation at the end of the prediction horizon. Dim=No
		target_action_norm: Target actions. The cost relative to actions will be 0 if the nromalized actions match this. Dim=Na
		weight_action: weight attributed for each action in the cost. Dim=Na
		exploration_factor: bonus attributed to the uncertainty of the predicted cost function. 
		use_constraints: if set to True, constraints for states will be used for the cost function
		state_min: minimum bound of the constraints for the normalized observation if use-constraints is True. Dim=No
		state_max: maximum bound of the constraints for the normalized observation if use-constraints is True. Dim=No
		area_multiplier: When including constraints in the cost function. The area of the predicted states outside the allowed region is used as penalty. This parameter multiply this area to adjust the effect of constraints on the cost
		clip_lower_bound_cost_to_0: if set to True, the cost will be clipped to 0. Should always be False

		Na: dimension of the env actions
		No: dimension of the env observations
		"""
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

		self.clip_lower_bound_cost_to_0 = clip_lower_bound_cost_to_0
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