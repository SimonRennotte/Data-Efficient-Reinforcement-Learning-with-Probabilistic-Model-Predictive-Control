from rl_gp_mpc.config_classes.config import Config
from rl_gp_mpc.config_classes.controller_config import ControllerConfig
from rl_gp_mpc.config_classes.actions_config import ActionsConfig
from rl_gp_mpc.config_classes.reward_config import RewardConfig
from rl_gp_mpc.config_classes.observation_config import ObservationConfig
from rl_gp_mpc.config_classes.memory_config import MemoryConfig
from rl_gp_mpc.config_classes.model_config import ModelConfig
from rl_gp_mpc.config_classes.training_config import TrainingConfig


def get_config(action_dim=1, state_dim=2, num_repeat_actions=5, len_horizon=10, include_time_model=False):
	observation_config = ObservationConfig(
		obs_var_norm = [1e-6, 1e-6]
	)

	reward_config = RewardConfig(
		target_state_norm=[1, 0.5], 

		weight_state=[1, 0], 
		weight_state_terminal=[3, 0], 

		target_action_norm=[0.5], 
		weight_action=[0.05],

		exploration_factor=2,

		use_constraints=False,
		state_min=[0.1, -0.1],
		state_max=[1.1, 1.1],
		area_multiplier=1, 
	)

	actions_config = ActionsConfig(
		limit_action_change=False,
		max_change_action_norm=[0.05],
		action_dim=action_dim,
		len_horizon=len_horizon
	)

	model_config = ModelConfig(
		gp_init = {
            "noise_covar.noise": [1e-4, 1e-4],
            "base_kernel.lengthscale": [0.75, 0.75, 0.75],
            "outputscale": [5e-2, 5e-2, 5e-2]
		},
		min_std_noise=1e-3,
		max_std_noise=3e-1,
		min_outputscale=1e-5,
		max_outputscale=0.95,
		min_lengthscale=4e-3,
		max_lengthscale=25.0,
		min_lengthscale_time=10,
		max_lengthscale_time=10000,
		init_lengthscale_time=100,
		include_time_model=include_time_model,
		state_dim=state_dim,
		action_dim=action_dim,
	)

	memory_config = MemoryConfig(
		min_error_prediction_state_for_memory=[3e-4, 3e-4],
		min_prediction_state_std_for_memory=[3e-3, 3e-3],
		points_batch_memory=1500
	)

	training_config = TrainingConfig(
		lr_train=7e-3,
		iter_train=15 ,
		training_frequency=10,
		clip_grad_value=1e-3,
		print_train=False,
		step_print_train=5
	)

	controller_config = ControllerConfig(
		len_horizon=len_horizon,
		num_repeat_actions=num_repeat_actions,
		clip_lower_bound_cost_to_0=True,
		actions_optimizer_params = {
			"disp": None, "maxcor": 8, "ftol": 1e-18, "gtol": 1e-18, "eps": 1e-2, "maxfun": 8,
			"maxiter": 8, "iprint": -1, "maxls": 8, "finite_diff_rel_step": None
		},
		init_from_previous_actions=True,
		restarts_optim=1,
		optimize=True
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


