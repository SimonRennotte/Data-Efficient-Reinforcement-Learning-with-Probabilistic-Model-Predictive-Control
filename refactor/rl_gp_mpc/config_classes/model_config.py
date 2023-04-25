from .utils.functions_process_config import convert_dict_lists_to_dict_tensor, extend_lengthscale_dim


class ModelConfig:
	def __init__(
		self,
		gp_init = {
			"noise_covar.noise": [1e-4, 1e-4, 1e-4], # variance = (std)²
			"base_kernel.lengthscale": [[0.75, 0.75, 0.75, 0.75],
										[0.75, 0.75, 0.75, 0.75],
										[0.75, 0.75, 0.75, 0.75]],
			"outputscale": [5e-2, 5e-2, 5e-2]
		},
		include_time_model=False,
		init_lengthscale_time=100,
		min_std_noise=1e-3,
		max_std_noise=3e-1 ,
		min_outputscale=1e-5,
		max_outputscale=0.95,
		min_lengthscale=4e-3,
		max_lengthscale=25.0,
		min_lengthscale_time=10,
		max_lengthscale_time=10000,
		state_dim=3,
		action_dim=1
	):
		self.include_time_model = include_time_model

		self.min_std_noise = min_std_noise
		self.max_std_noise = max_std_noise
		self.min_outputscale = min_outputscale
		self.max_outputscale = max_outputscale
		self.min_lengthscale = min_lengthscale
		self.max_lengthscale = max_lengthscale
		self.min_lengthscale_time = min_lengthscale_time
		self.max_lengthscale_time = max_lengthscale_time
		self.include_time_model = include_time_model

		self.gp_init = convert_dict_lists_to_dict_tensor(gp_init)

		if include_time_model:
			num_inputs = state_dim + action_dim + 1
			self.min_lengthscale = extend_lengthscale_dim(state_dim, num_inputs, self.min_lengthscale, self.min_lengthscale_time)
			self.max_lengthscale = extend_lengthscale_dim(state_dim, num_inputs, self.max_lengthscale, self.max_lengthscale_time)

			self.gp_init["base_kernel.lengthscale"] = extend_lengthscale_dim(
				num_models=state_dim, 
				num_inputs=state_dim+action_dim+1, 
				lengthscale=self.gp_init["base_kernel.lengthscale"], 
				lengthscale_time=init_lengthscale_time
			)