from .utils.functions_process_config import convert_dict_lists_to_dict_tensor, extend_dim, extend_dim_lengthscale_time


class ModelConfig:
	def __init__(
		self,
		gp_init:dict = {
			"noise_covar.noise": [1e-4, 1e-4, 1e-4], # variance = (std)²
			"base_kernel.lengthscale": [[0.75, 0.75, 0.75, 0.75],
										[0.75, 0.75, 0.75, 0.75],
										[0.75, 0.75, 0.75, 0.75]],
			"outputscale": [5e-2, 5e-2, 5e-2]
		},
		init_lengthscale_time:float=100,
		min_std_noise:float=1e-3,
		max_std_noise:float=3e-1,
		min_outputscale:float=1e-5,
		max_outputscale:float=0.95,
		min_lengthscale:float=4e-3,
		max_lengthscale:float=25.0,
		min_lengthscale_time:float=10,
		max_lengthscale_time:float=10000, 
		include_time_model:bool=False,
	):
		"""
		include_time_model: if set to true, the time of the observation (index of control) will be an additional contextual input to the model, which will allow to weight recent points more in the case that the environment changes over time
		gp_init: dict containing the initialization of the gp model parameters. The dimension are the number of observation, since there is 1 model per state/observation. Lengthcales dimensions are (No, No+Na). No: numver of ovservations, Na: number of actions
		All parameters are for normalized observations expect for the time lengthscales
		The noise parameter represents the noise variance. Take the sqrt to get the corresponding std.
		"""
		self.include_time_model = include_time_model

		self.min_std_noise = min_std_noise
		self.max_std_noise = max_std_noise
		self.min_outputscale = min_outputscale
		self.max_outputscale = max_outputscale
		self.min_lengthscale = min_lengthscale
		self.max_lengthscale = max_lengthscale
		self.min_lengthscale_time = min_lengthscale_time
		self.max_lengthscale_time = max_lengthscale_time
		self.init_lengthscale_time = init_lengthscale_time
		self.include_time_model = include_time_model

		self.gp_init = convert_dict_lists_to_dict_tensor(gp_init)

	def extend_dimensions_params(self, dim_state, dim_input):
		self.min_std_noise = extend_dim(self.min_std_noise, dim=(dim_state,))
		self.max_std_noise = extend_dim(self.max_std_noise, dim=(dim_state,))
		self.min_outputscale = extend_dim(self.min_outputscale, dim=(dim_state,))
		self.max_outputscale = extend_dim(self.max_outputscale, dim=(dim_state,))
		self.gp_init["noise_covar.noise"] = extend_dim(self.gp_init["noise_covar.noise"], dim=(dim_state,))
		self.gp_init["outputscale"] = extend_dim(self.gp_init["outputscale"], dim=(dim_state,))

		if self.include_time_model:
			self.min_lengthscale = extend_dim_lengthscale_time(self.min_lengthscale, self.min_lengthscale_time, dim_state, dim_input)
			self.max_lengthscale = extend_dim_lengthscale_time(self.max_lengthscale, self.max_lengthscale_time, dim_state, dim_input)

			self.gp_init["base_kernel.lengthscale"] = extend_dim_lengthscale_time(
				lengthscale=self.gp_init["base_kernel.lengthscale"], 
				lengthscale_time=self.init_lengthscale_time,
				num_models=dim_state, 
				num_inputs=dim_input, 
			)
		else:
			self.min_lengthscale = extend_dim(self.min_lengthscale, dim=(dim_state, dim_input))
			self.max_lengthscale = extend_dim(self.max_lengthscale, dim=(dim_state, dim_input))
			self.gp_init["base_kernel.lengthscale"] = extend_dim(self.gp_init["base_kernel.lengthscale"], dim=(dim_state, dim_input))