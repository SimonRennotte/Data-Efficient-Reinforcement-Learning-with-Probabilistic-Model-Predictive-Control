import torch


class AbstractStateRewardMapper:
	def __init__(self, config_reward):
		self.config = config_reward

	def get_reward_state(self, state:torch.Tensor, action:torch.Tensor):
		raise NotImplementedError

	def get_reward_state_terminal(self, state:torch.Tensor):
		raise NotImplementedError

	def get_reward_trajectory(self, states:torch.Tensor, actions:torch.Tensor):
		raise NotImplementedError


class StateRewardMapper:
	def __init__(self, config_reward):
		super().__init__(config_reward)

	def get_reward_state(self, state: torch.Tensor, action:torch.Tensor) -> float:
		pass

	def get_reward_state_terminal(self, state:torch.Tensor) -> float:
		pass

	def get_rewards_trajectory(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
		pass

	def get_reward_trajectory(self, states:torch.Tensor, actions:torch.Tensor) -> float:
		pass
