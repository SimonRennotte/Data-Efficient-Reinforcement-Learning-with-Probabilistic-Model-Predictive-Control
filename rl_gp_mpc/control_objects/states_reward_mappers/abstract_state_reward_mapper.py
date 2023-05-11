import torch

from rl_gp_mpc.config_classes.reward_config import RewardConfig

class AbstractStateRewardMapper:
	def __init__(self, config:RewardConfig):
		self.config = config

	def get_reward(self, state_mu:torch.Tensor, state_var: torch.Tensor, action:torch.Tensor):
		raise NotImplementedError

	def get_reward_terminal(self, state_mu:torch.Tensor, state_var:torch.Tensor):
		raise NotImplementedError

	def get_rewards_trajectory(self, states_mu:torch.Tensor, states_var:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

