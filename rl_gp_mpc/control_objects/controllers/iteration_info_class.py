import numpy as np
import torch

NUM_DECIMALS_REPR = 3

class IterationInformation:
	def __init__(self,
		iteration:int,
		state:float,
		cost:float,
		cost_std:float,
		mean_predicted_cost:float,
		mean_predicted_cost_std:float,
		lower_bound_mean_predicted_cost:float,
		predicted_idxs:np.array,
		predicted_states:np.array,
		predicted_states_std:np.array,
		predicted_actions:np.array,
		predicted_costs:np.array,
		predicted_costs_std:np.array,
	):
		self.iteration=iteration
		self.state=state
		self.cost=cost
		self.cost_std=cost_std
		self.mean_predicted_cost=mean_predicted_cost
		self.mean_predicted_cost_std=mean_predicted_cost_std
		self.lower_bound_mean_predicted_cost=lower_bound_mean_predicted_cost
		self.predicted_idxs=predicted_idxs
		self.predicted_states=predicted_states
		self.predicted_states_std=predicted_states_std
		self.predicted_actions=predicted_actions
		self.predicted_costs=predicted_costs
		self.predicted_costs_std=predicted_costs_std

	def to_arrays(self):
		for key in self.__dict__.keys():
			if isinstance(self.__dict__[key], torch.Tensor):
				self.__setattr__(key, np.array(self.__dict__[key]))

	def to_tensors(self):
		for key in self.__dict__.keys():
			if isinstance(self.__dict__[key], np.array):
				self.__setattr__(key, torch.Tensor(self.__dict__[key]))

	def __str__(self):
		np.set_printoptions(precision=NUM_DECIMALS_REPR, suppress=True)
		iter_info_dict = self.__dict__
		str_rep = "\n"
		for key, item in iter_info_dict.items():
			#if key in ['predicted_states', 'predicted_states_std']:
			#	continue
			if isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
				item = np.array2string(np.array(item), threshold=np.inf, max_line_width=np.inf, separator=',').replace('\n', '') # np.round( ), NUM_DECIMALS_REPR)# .tolist()
			else:
				item = np.round(item, NUM_DECIMALS_REPR)
			str_rep += f"{key}: {item}\n"
		return str_rep