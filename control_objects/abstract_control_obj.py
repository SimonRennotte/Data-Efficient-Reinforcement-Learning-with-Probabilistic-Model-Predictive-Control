import torch


class BaseControllerObject:
	def __init__(self, observation_space, action_space, n_points_init_memory=1000):
		self.action_space = action_space
		self.obs_space = observation_space
		self.num_states = self.obs_space.shape[0]
		self.num_actions = action_space.shape[0]
		self.num_inputs = self.num_states + self.num_actions
		self.points_add_mem_when_full = n_points_init_memory
		self.len_mem = 0

		self.x = torch.empty(n_points_init_memory, self.num_inputs)
		self.y = torch.empty(n_points_init_memory, self.num_states)
		self.rewards = torch.empty(n_points_init_memory)

	def add_memory(self, observation, action, new_observation, reward, **kwargs):
		raise NotImplementedError()

	def compute_action(self, observation, s_observation):
		raise NotImplementedError()

	def train(self):
		raise NotImplementedError()