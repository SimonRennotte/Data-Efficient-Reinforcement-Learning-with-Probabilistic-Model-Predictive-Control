import torch


class BaseControllerObject:
	def __init__(self, observation_space, action_space, n_points_init_memory=1000):
		self.action_space = action_space
		self.observation_space = observation_space
		self.num_outputs = self.observation_space.shape[0]
		self.num_inputs = self.observation_space.shape[0] + action_space.shape[0]
		self.x = torch.empty(n_points_init_memory, self.num_inputs)
		self.y = torch.empty(n_points_init_memory, self.num_outputs)
		self.num_points_add_memory_when_full = 1000
		self.num_points_memory = 0

	def add_point_memory(self, observation, action, new_observation, reward, **kwargs):
		raise NotImplementedError()

	def compute_prediction_action(self, observation, s_observation):
		raise NotImplementedError()

	def train(self):
		raise NotImplementedError()

	def save_plot_model_3d(self):
		raise NotImplementedError

	def save_plot_history(self):
		raise NotImplementedError