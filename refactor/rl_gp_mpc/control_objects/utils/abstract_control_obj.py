import torch
from rl_gp_mpc.config_classes.config import Config


class BaseControllerObject:
	def __init__(self, observation_space, action_space, config: Config):
		self.action_space = action_space
		self.obs_space = observation_space
		self.num_states = self.obs_space.shape[0]
		self.num_actions = action_space.shape[0]
		self.num_inputs = self.num_states + self.num_actions

		if config.model.include_time_model:
			self.num_inputs += 1

		self.x = torch.empty(config.memory.points_batch_memory, self.num_inputs)
		self.y = torch.empty(config.memory.points_batch_memory, self.num_states)
		self.rewards = torch.empty(config.memory.points_batch_memory)

		self.len_mem = 0

	def add_memory(self, observation, action, new_observation, reward, **kwargs):
		raise NotImplementedError()

	def compute_action(self, observation, s_observation):
		raise NotImplementedError()

	def train(self):
		raise NotImplementedError()