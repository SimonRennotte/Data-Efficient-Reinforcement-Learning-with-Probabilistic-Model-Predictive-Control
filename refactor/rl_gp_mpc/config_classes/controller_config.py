class ControllerConfig:
	def __init__(self, 
		len_horizon=15, 
		num_repeat_actions=1, 
		clip_lower_bound_cost_to_0=False, 
		actions_optimizer_params={
			"disp": None, "maxcor": 30, "ftol": 1e-99, "gtol": 1e-99, "eps": 1e-2, "maxfun": 30,
			"maxiter": 30, "iprint": -1, "maxls": 30, "finite_diff_rel_step": None
		},
		init_from_previous_actions=True,
		restarts_optim=1,
		optimize=True
	):
		self.len_horizon = len_horizon
		self.num_repeat_actions = num_repeat_actions
		self.clip_lower_bound_cost_to_0 = clip_lower_bound_cost_to_0
		self.actions_optimizer_params = actions_optimizer_params
		self.init_from_previous_actions = init_from_previous_actions
		self.restarts_optim=restarts_optim
		self.optimize = optimize