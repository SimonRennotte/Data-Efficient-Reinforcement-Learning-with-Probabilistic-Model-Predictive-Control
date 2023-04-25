import torch

from rl_gp_mpc.config_classes.reward_config import RewardConfig
from rl_gp_mpc.config_classes.observation_config import ObservationConfig


def calculate_factorizations(x, y, models):
    """
    Compute iK and beta using the points in memory, which are needed to make predictions with the gaussian processes.
    These two variables only depends on data in memory, and not on input distribution,
    so they separated from other computation such that they can be computed outside the optimisation function,
    which is computed multiple times at each iteration

    Function inspired from
    https://github.com/nrontsis/PILCO/blob/6a962c8e4172f9e7f29ed6e373c4be2dd4b69cb7/pilco/models/mgpr.py#L81,
    reimplemented from tensorflow to pytorch
    Args:
        x (torch.Tensor): matrix containing the states and actions. Dim=(Nm, Ns + Na)
        y (torch.Tensor): matrix containing the states change. Dim=(Nm, Ns)
        models (list of gpytorch.models.ExactGP): list containing the gp models used to predict each state change.
                                                    Len=Ns
        Ns: number of states
        Na: number of actions
        Nm: number of points in memory

    Returns:
        iK (torch.Tensor): needed by the gaussian processes models to compute the predictions
        beta (torch.Tensor): needed by the gaussian processes models to compute the predictions

    """
    K = torch.stack([model.covar_module(x).evaluate() for model in models])
    batched_eye = torch.eye(K.shape[1]).repeat(K.shape[0], 1, 1)
    L = torch.cholesky(K + torch.stack([model.likelihood.noise for model in models])[:, None] * batched_eye)
    iK = torch.cholesky_solve(batched_eye, L)
    Y_ = (y.t())[:, :, None]
    beta = torch.cholesky_solve(Y_, L)[:, :, 0]
    return iK, beta

def compute_cost_unnormalized(obs, action, low_obs, high_obs, low_action, high_action, config_reward: RewardConfig, config_observation:ObservationConfig, obs_var=None):
    """
    Compute the cost on un-normalized state and actions.
    Takes in numpy array and returns numpy array.
    Meant to be used to compute the cost outside the object.
    Args:
        obs (numpy.array): state (or observation). shape=(Ns,)
        action (numpy.array): action. Shape=(Na,)
        obs_var (numpy.array): state (or observation) variance. Default=None. shape=(Ns, Ns)
                                If set to None, the observation constant stored inside the object will be used

    Returns:
        cost_mu (float): Mean of the cost
        cost_var (float): variance of the cost
    """
    obs_norm = to_normed_obs_tensor(obs, low=low_obs, high=high_obs)
    action_norm = to_normed_action_tensor(action, low=low_action, high=high_action)
    if obs_var is None:
        obs_var_norm = config_observation.obs_var_norm
    else:
        obs_var_norm = to_normed_var_tensor(obs_var, low=low_obs, high=high_obs)
    cost_mu, cost_var = compute_cost(obs_norm, obs_var_norm, action_norm, config_reward)
    return cost_mu.item(), cost_var.item()
    
    
def compute_cost(state_mu, state_var, action, config_reward: RewardConfig):
    """
    Compute the quadratic cost of one state distribution or a trajectory of states distributions
    given the mean value and variance of states (observations), the weight matrix, and target state.
    The state, state_var and action must be normalized.
    If reading directly from the gym env observation,
    this can be done with the gym env action space and observation space.
    See an example of normalization in the add_points_memory function.
    Args:
        state_mu (torch.Tensor): normed mean value of the state or observation distribution
                                (elements between 0 and 1). dim=(Ns) or dim=(Np, Ns)
        state_var (torch.Tensor): normed variance matrix of the state or observation distribution
                                    (elements between 0 and 1)
                                    dim=(Ns, Ns) or dim=(Np, Ns, Ns)
        action (torch.Tensor): normed actions. (elements between 0 and 1).
                                dim=(Na) or dim=(Np, Na)

        Np: length of the prediction trajectory. (=self.len_horizon)
        Na: dimension of the gym environment actions
        Ns: dimension of the gym environment states

    Returns:
        cost_mu (torch.Tensor): mean value of the cost distribution. dim=(1) or dim=(Np)
        cost_var (torch.Tensor): variance of the cost distribution. dim=(1) or dim=(Np)
    """

    if state_var.ndim == 3:
        error = torch.cat((state_mu, action), 1) - config_reward.target_state_action_norm
        state_action_var = torch.cat((
            torch.cat((state_var, torch.zeros((state_var.shape[0], state_var.shape[1], action.shape[1]))), 2),
            torch.zeros((state_var.shape[0], action.shape[1], state_var.shape[1] + action.shape[1]))), 1)
    else:
        error = torch.cat((state_mu, action), 0) - config_reward.target_state_action_norm
        state_action_var = torch.block_diag(state_var, torch.zeros((action.shape[0], action.shape[0])))
    cost_mu = torch.diagonal(torch.matmul(state_action_var, config_reward.weight_matrix_cost),
            dim1=-1, dim2=-2).sum(-1) + \
            torch.matmul(torch.matmul(error[..., None].transpose(-1, -2), config_reward.weight_matrix_cost),
                    error[..., None]).squeeze()
    TS = config_reward.weight_matrix_cost @ state_action_var
    cost_var_term1 = torch.diagonal(2 * TS @ TS, dim1=-1, dim2=-2).sum(-1)
    cost_var_term2 = TS @ config_reward.weight_matrix_cost
    cost_var_term3 = (4 * error[..., None].transpose(-1, -2) @ cost_var_term2 @ error[..., None]).squeeze()
    cost_var = cost_var_term1 + cost_var_term3
    if config_reward.use_constraints:
        if state_mu.ndim == 2:
            state_distribution = [torch.distributions.normal.Normal(state_mu[idx], state_var[idx]) for idx in
                range(state_mu.shape[0])]
            penalty_min_constraint = torch.stack([state_distribution[time_idx].cdf(torch.Tensor(
                config_reward.state_min)) * config_reward.area_multiplier for
                time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
            penalty_max_constraint = torch.stack([(1 - state_distribution[time_idx].cdf(torch.Tensor(
                config_reward.state_max))) * config_reward.area_multiplier for
                time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
        else:
            state_distribution = torch.distributions.normal.Normal(state_mu, state_var)
            penalty_min_constraint = ((state_distribution.cdf(torch.Tensor(
                config_reward.state_min))) * config_reward.area_multiplier).diagonal(0, -1, -2).sum(-1)
            penalty_max_constraint = ((1 - state_distribution.cdf(
                torch.Tensor(config_reward.state_max))) * config_reward.area_multiplier).diagonal(0, -1, -2).sum(-1)
        cost_mu = cost_mu + penalty_max_constraint + penalty_min_constraint

    return cost_mu, cost_var

def compute_cost_terminal(state_mu, state_var, config_reward:RewardConfig):
    """
    Compute the terminal cost of the prediction trajectory.
    Args:
        state_mu (torch.Tensor): mean value of the terminal state distribution. dim=(Ns)
        state_var (torch.Tensor): variance matrix of the terminal state distribution. dim=(Ns, Ns)

    Returns:
        cost_mu (torch.Tensor): mean value of the cost distribution. dim=(1)
        cost_var (torch.Tensor): variance of the cost distribution. dim=(1)
    """
    error = state_mu - config_reward.target_state_norm
    cost_mu = torch.trace(torch.matmul(state_var, config_reward.weight_matrix_cost_terminal)) + \
                torch.matmul(torch.matmul(error.t(), config_reward.weight_matrix_cost_terminal), error)
    TS = config_reward.weight_matrix_cost_terminal @ state_var
    cost_var_term1 = torch.trace(2 * TS @ TS)
    cost_var_term2 = 4 * error.t() @ TS @ config_reward.weight_matrix_cost_terminal @ error
    cost_var = cost_var_term1 + cost_var_term2
    return cost_mu, cost_var

def to_normed_obs_tensor(obs, low, high):
    """
    Compute the norm of observation using the min and max of the observation_space of the gym env.

    Args:
        obs  (numpy.array): observation from the gym environment. dim=(Ns,)

    Returns:
        state_mu_norm (torch.Tensor): normed states
    """
    state_mu_norm = torch.Tensor((obs - low) / (high - low))
    return state_mu_norm

def to_normed_var_tensor(obs_var, low, high):
    """
    Compute the norm of the observation variance matrix using
    the min and max of the observation_space of the gym env.

    Args:
        obs_var  (numpy.array): unnormalized variance of the state. dim=(Ns,)

    Returns:
        obs_var_norm (torch.Tensor): normed variance of the state
    """
    obs_var_norm = obs_var / (high - low)
    obs_var_norm = torch.Tensor(obs_var_norm / (high - low).T)
    return obs_var_norm

def to_normed_action_tensor(action, low, high) -> torch.Tensor:
    """
    Compute the norm of the action using the min and max of the action_space of the gym env.

    Args:
        action  (numpy.array): un-normalized action. dim=(Na,)
                                Na: dimension of action_space
    Returns:
        action_norm (torch.Tensor): normed action

    """
    action_norm = torch.Tensor((action - low) / (high - low))
    return action_norm

def denorm_action(action_norm, low, high):
    """
    Denormalize the action using the min and max of the action_space of the gym env, so that
    it can be apllied on the gym env

    Args:
        action_norm  (numpy.array or torch.Tensor): normed action. dim=(Na,)
                                                Na: dimension of action_space
    Returns:
        action (numpy_array or torch.Tensor): un-normalised action. dim=(Na,)
                                                    Na: dimension of action_space
    """
    action = action_norm * (high - low) + low
    return action
