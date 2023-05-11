class ControllerConfig:
	def __init__(self, 
		len_horizon:int=15, 
		actions_optimizer_params:dict={
			"disp": None, "maxcor": 30, "ftol": 1e-99, "gtol": 1e-99, "eps": 1e-2, "maxfun": 30,
			"maxiter": 30, "iprint": -1, "maxls": 30, "finite_diff_rel_step": None
		},
		init_from_previous_actions:bool=True,
		restarts_optim:int=1,
		optimize:bool=True,
		num_repeat_actions:int=1
	):
		self.len_horizon = len_horizon
		self.actions_optimizer_params = actions_optimizer_params
		self.init_from_previous_actions = init_from_previous_actions
		self.restarts_optim=restarts_optim
		self.optimize = optimize
		self.num_repeat_actions=num_repeat_actions