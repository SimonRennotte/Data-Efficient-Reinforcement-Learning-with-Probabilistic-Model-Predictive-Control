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
		"""
		len_horizon: number of timesteps used by the mpc to find the optimal action. The effective number of predicted timtesteps will be len_horizon * num_repeat_actions
		actions_optimizer_params: parameters of the optimizer used. Refer to the scipy documentation for more information about the parameters
		init_from_previous_actions: if set to true, the actions of the optimizer will be initialized with actions of the previous timestep, shifted by 1
		restarts_optim: number of restarts for the optimization. Can help escape local mimimum of the actions if init_from_previous_actions is set to true
		optimize: if set to false, the optimal actions will be taken at random. Used for debugging only
		num_repeat_actions: The actions will be repeated this number of time. It allows to predict more timesteps in advance while keeping the same len_horizon number. It can also be used to be more resilient to noise with the model. The model only predicts the change after all actions have been repeated.
		"""
		self.len_horizon = len_horizon
		self.actions_optimizer_params = actions_optimizer_params
		self.init_from_previous_actions = init_from_previous_actions
		self.restarts_optim=restarts_optim
		self.optimize = optimize
		self.num_repeat_actions=num_repeat_actions