
### Json parameters to be specified for each gym environment
It is important for the selection of parameters that the input to the gaussian processes are normalized between 0 and 1.
Also, the gaussian processes predict changes in states and not states directly.

- gp_init: initial values of the hyperparameters of the gaussian processes
    - noise_covar.noise: variance of predictions of the gaussian processes at known points (dim=(number of states))
	- base_kernel.lengthscale: lengthscales for each gaussian process (dim=(number of states, number of input) or dim=(number of states))
	- outputscale: scale of the gaussian processes (dim=(number of states)))
	
- gp_constraints: constraints on the hyperparameters of the gaussian processes
    - min_std_noise: minimum value of the standard deviation of the gp noise (dim=scalar)
    - max_std_noise: maximum value of the standard deviation of the gp noise (dim=scalar)
    - min_outputscale: minimum value of the outputscales (dim=scalar)
    - max_outputscale: maximum value of the outputscales (dim=scalar)
    - min_lengthscale: minimum value of the lengthscales (dim=scalar)
    - max_lengthscale: maximum value of the lengthscales (dim=scalar)

- controller: parameters relative to the cost function and MPC
    - target_state_norm: value of the normed states to attain to minimize the cost (dim=(dimension of states))
    - weight_state: weights of each state dimension in the cost function (dim=(dimension of states))
    - weight_state_terminal: weights of each state dimension in the terminal cost function, at the end of the prediction horizon (dim=(dimension of states))
    - target_action_norm: value of the normed actions to attain to minimize the cost (dim=(dimension of actions))
    - weight_action: weights of each action dimension in the cost function (dim=(dimension of actions))
    - obs_var_norm: variance of the observations of states, if there is observation noise (dim=(dimension of states)). Do not set too low or errors might occur.
    - len_horizon: length of the horizon used to find the optimal actions. The total horizon length in time steps = len_horizon * num_repeat_actions (dim=scalar)
    - num_repeat_actions: number of time steps to repeat the planned actions
    - exploration_factor: the value to be minimized is (sum of predicted cost - exploration_factor * sum of the predicted cost uncertainty). A higher value will lead to more exploration (dim=scalar)
    - limit_action_change: if set to true, the variation of the normalized actions will be limited, from time step to time step by the parameter max_derivative_actions_norm (dim=scalar)
    - max_change_action_norm: limitation on the variation of normalized actions from on control step to another if limit_derivative_actions is set to 1. (dim=(dimension of actions))
    - clip_lower_bound_cost_to_0: if set to true, the optimized cost (with exploration parameter) will be clipped to 0 if negative.

- params_constraints_states:
     - use_constraints: if set to true, the constraints will be used, if set to 0, it will be ignored
     - state_min: minimum allowed value of the states (dim=(number of states))
     - state_max: maximum allowed value of the states (dim=(number of states))
     - area_multiplier: At the moment, constraints on the states are added as a penalty in the predicted cost trajectory. The value of this penalty is the area of the predicted state distribution that violate the constraints. This penalty is multiplied by this parameter

- params_train: parameters used for the training of the gaussian processes hyper parameters, done in a parallel process
    - lr_train: learning rate
    - n_iter_train: number of training iteration
    - training_frequency: training will occur every time a certain number of points have been added in the gps memory. This parameter set the frequency of training process in number of points added in memory.
    - clip_grad_value: maximum value of the gradient of the trained parameters, for more stability
    - print_train: if set to true, the values optimized will be printed during training
    - step_print_train: if print_train is set to 1, this parameter specifies the frequency of printing during training

- params_actions_optimizer: parameters of the optimizer used to find the optimal actions by minimizing the lower bound of the predicted cost. See the scipy documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html?highlight=minimize Note: the jacobian is used for optimization, so the parameters eps is not used.

- params_memory: parameters relative to memory storage. Every point is not stored in memory. Either the error of the prediction must be above a threshold or the standard deviation of the prediction uncertainty must be above a threshold
    - min_error_prediction_state_for_memory: minimum prediction error for each of the predicted states (dim=(number of states))
    - min_prediction_state_std_for_memory: minimum predicted standard deviation for each of the predicted states (dim=(number of states))