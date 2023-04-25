from rl_gp_mpc.config_classes.config import Config
from rl_gp_mpc.config_classes.controller_config import ControllerConfig
from rl_gp_mpc.config_classes.actions_config import ActionsConfig
from rl_gp_mpc.config_classes.reward_config import RewardConfig
from rl_gp_mpc.config_classes.observation_config import ObservationConfig
from rl_gp_mpc.config_classes.memory_config import MemoryConfig
from rl_gp_mpc.config_classes.model_config import ModelConfig
from rl_gp_mpc.config_classes.training_config import TrainingConfig


def get_config(action_dim=2, state_dim=2, num_repeat_actions=1, len_horizon=4, include_time_model=True):
	observation_config = ObservationConfig(
		obs_var_norm = [1e-6, 1e-6]
	)

	reward_config = RewardConfig(
		target_state_norm=[0.5, 0.5], 

		weight_state=[1, 1], 
		weight_state_terminal=[1, 1], 

		target_action_norm=[0, 0], 
		weight_action=[1e-3, 1e-3],

		exploration_factor=1,

		use_constraints=False,
		state_min=[0.1, 0.3],
		state_max=[0.9, 0.8],
		area_multiplier=1, 
	)

	actions_config = ActionsConfig(
		limit_action_change=False,
		max_change_action_norm=[0.1, 0.1],
		action_dim=action_dim,
		len_horizon=len_horizon
	)

	model_config = ModelConfig(
		gp_init = {
            "noise_covar.noise": [1e-4, 1e-4],
            "base_kernel.lengthscale": [0.75, 0.75],
            "outputscale": [5e-2, 5e-2]
		},
		init_lengthscale_time=100,
		min_std_noise=1e-4,
		max_std_noise=3e-1,
		min_outputscale=1e-5,
		max_outputscale=0.95,
		min_lengthscale=4e-3,
		max_lengthscale=25.0,
		include_time_model=include_time_model,
		min_lengthscale_time=5,
		max_lengthscale_time=1000,
		state_dim=state_dim,
		action_dim=action_dim,
	)

	memory_config = MemoryConfig(
		min_error_prediction_state_for_memory=[1e-5, 1e-5],
		min_prediction_state_std_for_memory=[3e-3, 3e-3],
		points_batch_memory=1500
	)

	training_config = TrainingConfig(
		lr_train=7e-3,
		iter_train=15,
		training_frequency=30,
		clip_grad_value=1e-3,
		print_train=False,
		step_print_train=5
	)

	controller_config = ControllerConfig(
		len_horizon=len_horizon,
		num_repeat_actions=num_repeat_actions,
		clip_lower_bound_cost_to_0=False,
		actions_optimizer_params = {
			"disp": None, "maxcor": 6, "ftol": 1e-99, "gtol": 1e-99, "eps": 1e-2, "maxfun": 6,
			"maxiter": 6, "iprint": -1, "maxls": 6, "finite_diff_rel_step": None
		}
	)

	config = Config(
		observation_config=observation_config,
		reward_config=reward_config,
		actions_config=actions_config,
		model_config=model_config,
		memory_config=memory_config,
		training_config=training_config,
		controller_config=controller_config, 
	)
	return config


