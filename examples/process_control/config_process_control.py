from rl_gp_mpc.config_classes.total_config import Config
from rl_gp_mpc.config_classes.controller_config import ControllerConfig
from rl_gp_mpc.config_classes.actions_config import ActionsConfig
from rl_gp_mpc.config_classes.reward_config import RewardConfig
from rl_gp_mpc.config_classes.observation_config import ObservationConfig
from rl_gp_mpc.config_classes.memory_config import MemoryConfig
from rl_gp_mpc.config_classes.model_config import ModelConfig
from rl_gp_mpc.config_classes.training_config import TrainingConfig


def get_config(len_horizon=4, include_time_model=True, num_repeat_actions=10):
	observation_config = ObservationConfig(
		obs_var_norm = [1e-6, 1e-6]
	)

	reward_config = RewardConfig(
		target_state_norm=[0.5, 0.5], 

		weight_state=[1, 1], 
		weight_state_terminal=[1, 1], 

		target_action_norm=[0, 0], 
		weight_action=[1e-4, 1e-4],

		exploration_factor=1,

		use_constraints=False,
		state_min=[0.1, 0.3],
		state_max=[0.9, 0.8],
		area_multiplier=1, 

		clip_lower_bound_cost_to_0=False,
	)

	actions_config = ActionsConfig(
		limit_action_change=False,
		max_change_action_norm=[0.1, 0.2]
	)

	model_config = ModelConfig(
		gp_init = {
            "noise_covar.noise": [1e-5, 1e-5],
            "base_kernel.lengthscale": [0.25, 0.25],
            "outputscale": [5e-2, 5e-2]
		},
		init_lengthscale_time=100,
		min_std_noise=1e-3,
		max_std_noise=3e-1,
		min_outputscale=1e-5,
		max_outputscale=0.95,
		min_lengthscale=5e-2,
		max_lengthscale=25.0,
		include_time_model=include_time_model,
		min_lengthscale_time=5,
		max_lengthscale_time=1000
	)

	memory_config = MemoryConfig(
		check_errors_for_storage=True,
		min_error_prediction_state_for_memory=[1e-5, 1e-5],
		min_prediction_state_std_for_memory=[3e-3, 3e-3],
		points_batch_memory=1500
	)

	training_config = TrainingConfig(
		lr_train=7e-3,
		iter_train=15,
		training_frequency=15,
		clip_grad_value=1e-3,
		print_train=False,
		step_print_train=5
	)

	controller_config = ControllerConfig(
		len_horizon=len_horizon,
		actions_optimizer_params = {
			"disp": None, "maxcor": 15, "ftol": 1e-99, "gtol": 1e-99, "eps": 1e-2, "maxfun": 15,
			"maxiter": 15, "iprint": -1, "maxls": 15, "finite_diff_rel_step": None
		},
		init_from_previous_actions=True,
		restarts_optim=2,
		optimize=True,
		num_repeat_actions=num_repeat_actions
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


