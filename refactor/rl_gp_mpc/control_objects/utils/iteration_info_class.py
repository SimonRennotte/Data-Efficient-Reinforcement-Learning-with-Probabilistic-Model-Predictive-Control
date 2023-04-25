import numpy as np

class IterationInformation:
	def __init__(self,
		iteration:int,
		state:float,
		predicted_states:np.array,
		predicted_states_std:np.array,
		predicted_actions:np.array,
		cost:float,
		cost_std:float,
		predicted_costs:np.array,
		predicted_costs_std:np.array,
		mean_predicted_cost:float,
		mean_predicted_cost_std:float,
		lower_bound_mean_predicted_cost:float
	):
		self.iteration=iteration
		self.state=state
		self.predicted_states=predicted_states
		self.predicted_states_std=predicted_states_std
		self.predicted_actions=predicted_actions
		self.cost=cost
		self.cost_std=cost_std
		self.predicted_costs=predicted_costs
		self.predicted_costs_std=predicted_costs_std
		self.mean_predicted_cost=mean_predicted_cost
		self.mean_predicted_cost_std=mean_predicted_cost_std
		self.lower_bound_mean_predicted_cost=lower_bound_mean_predicted_cost