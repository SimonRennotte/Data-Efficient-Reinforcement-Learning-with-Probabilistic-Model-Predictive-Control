import multiprocessing

import numpy as np
import torch
from scipy.optimize import minimize

from rl_gp_mpc.control_objects.actions_mappers.action_init_functions import generate_mpc_action_init_random, generate_mpc_action_init_frompreviousiter
from rl_gp_mpc.config_classes.total_config import Config
from rl_gp_mpc.control_objects.observations_states_mappers.normalization_observation_state_mapper import NormalizationObservationStateMapper
from rl_gp_mpc.control_objects.actions_mappers.derivative_action_mapper import DerivativeActionMapper
from rl_gp_mpc.control_objects.actions_mappers.normalization_action_mapper import NormalizationActionMapper
from rl_gp_mpc.control_objects.models.gp_model import GpStateTransitionModel
from rl_gp_mpc.control_objects.states_reward_mappers.setpoint_distance_reward_mapper import SetpointStateRewardMapper
from rl_gp_mpc.control_objects.memories.gp_memory import Memory
from rl_gp_mpc.control_objects.utils.pytorch_utils import Clamp

from .abstract_controller import BaseControllerObject
from .iteration_info_class import IterationInformation


class GpMpcController(BaseControllerObject):
    def __init__(self, observation_low:np.array, observation_high:np.array, action_low:np.array, action_high:np.array, config:Config):
        self.config = config
        self.observation_state_mapper = NormalizationObservationStateMapper(config=config.observation, observation_low=observation_low, observation_high=observation_high)

        if self.config.actions.limit_action_change:
            self.actions_mapper = DerivativeActionMapper(config=self.config.actions, action_low=action_low, action_high=action_high, len_horizon=self.config.controller.len_horizon)
        else:
            self.actions_mapper = NormalizationActionMapper(config=self.config.actions, action_low=action_low, action_high=action_high, len_horizon=self.config.controller.len_horizon)

        self.transition_model = GpStateTransitionModel(config=self.config.model, 
                                                        dim_state=self.observation_state_mapper.dim_observation, 
                                                        dim_action=self.actions_mapper.dim_action)
        self.state_reward_mapper = SetpointStateRewardMapper(config=self.config.reward)

        self.memory = Memory(self.config.memory, 
                            dim_input=self.transition_model.dim_input, dim_state=self.transition_model.dim_state, 
                            include_time_model=self.transition_model.config.include_time_model, 
                            step_model=self.config.controller.num_repeat_actions)

        self.actions_mpc_previous_iter = None
        self.clamp_lcb_class = Clamp()

        self.iter_ctrl = 0

        self.num_cores_main = multiprocessing.cpu_count()
        self.ctx = multiprocessing.get_context('spawn')
        self.queue_train = self.ctx.Queue()

        self.info_iters = {}

    def get_action(self, obs_mu:np.array, obs_var:np.array=None, random:bool=False):
        """
        Get the optimal action given the observation by optimizing
        the actions of the simulated trajectory with the gaussian process models such that the lower confidence bound of
        the mean cost of the trajectory is minimized.
        Only the first action of the prediction window is returned.

        Args:
            obs_mu (numpy.array): unnormalized observation from the gym environment. dim=(Ns)
            obs_var (numpy.array): unnormalized variance of the observation from the gym environment. dim=(Ns, Ns).
                                    default=None. If it is set to None,
                                    the observation noise from the json parameters will be used for every iteration.
                                    Ns is the dimension of states in the gym environment.

        Returns:
            next_action_raw (numpy.array): action to use in the gym environment.
                                        It is denormalized, so it can be used directly.
                                        dim=(Na), where Ns is the dimension of the action_space

        """
        # Check for parallel process that are open but not alive at each iteration to retrieve the results and close them
        self.check_and_close_processes()
        if (self.iter_ctrl % self.config.controller.num_repeat_actions) == 0:
            self.memory.prepare_for_model()
            with torch.no_grad():
                state_mu, state_var = self.observation_state_mapper.get_state(obs=obs_mu, obs_var=obs_var, update_internals=True)

            if random:
                actions_model = self._get_random_actions(state_mu, state_var)
            else:
                actions_model = self._get_optimal_actions(state_mu, state_var)
            actions_raw = self.actions_mapper.transform_action_model_to_action_raw(actions_model, update_internals=True)
            next_action_raw = actions_raw[0]

            # all the information fo the trajectory are stored in the object when using the function compute_mean_lcb_trajectory
            with torch.no_grad():
                reward, reward_var = self.state_reward_mapper.get_reward(state_mu, state_var, actions_model[0])
                states_std_pred = torch.diagonal(self.states_var_pred, dim1=-2, dim2=-1).sqrt()
                indexes_predicted = np.arange(self.iter_ctrl, self.iter_ctrl + self.config.controller.len_horizon * self.config.controller.num_repeat_actions, self.config.controller.num_repeat_actions)
                self.iter_info = IterationInformation(
                                    iteration=self.iter_ctrl,
                                    state=self.states_mu_pred[0],
                                    cost=-reward.item(),
                                    cost_std=reward_var.sqrt().item(),
                                    mean_predicted_cost=np.min([-self.rewards_trajectory.mean().item(), 3]),
                                    mean_predicted_cost_std=self.rewards_traj_var.sqrt().mean().item(),
                                    lower_bound_mean_predicted_cost=self.cost_traj_mean_lcb.item(),
                                    predicted_idxs=indexes_predicted,
                                    predicted_states=self.states_mu_pred,
                                    predicted_states_std=states_std_pred,
                                    predicted_actions=actions_model,
                                    predicted_costs=-self.rewards_trajectory,
                                    predicted_costs_std=self.rewards_traj_var.sqrt(),
                                )
                self.store_iter_info(self.iter_info)
            self.past_action = next_action_raw
        else:
            next_action_raw = self.past_action

        self.iter_ctrl += 1
        return np.array(next_action_raw)

    def _get_optimal_actions(self, state_mu:torch.Tensor, state_var:torch.Tensor) -> torch.Tensor:
        x_mem, y_mem = self.memory.get()
        with torch.no_grad():
            self.transition_model.prepare_inference(x_mem, y_mem)

        # The optimize function from the scipy library.
        # It is used to get the optimal actions_norm in the prediction window
        # that minimizes the lower bound of the predicted cost. The jacobian is used,
        # otherwise the computation times would be 5 to 10x slower (for the tests I used)
        opt_fun = np.inf
        actions_mpc_optim = None
        for idx_restart in range(self.config.controller.restarts_optim):
            # only init from previous actions for the first restart, if it's available in the object and specified to be used in the config
            if self.config.controller.init_from_previous_actions and (self.actions_mpc_previous_iter is not None) and (idx_restart == 0):
                actions_mpc_init = generate_mpc_action_init_frompreviousiter(self.actions_mpc_previous_iter, dim_action=self.actions_mapper.dim_action)
            else:
                actions_mpc_init = generate_mpc_action_init_random(len_horizon=self.config.controller.len_horizon, dim_action=self.actions_mapper.dim_action)

            if self.config.controller.optimize:
                iter_res = minimize(fun=self.compute_mean_lcb_trajectory,
                    x0=actions_mpc_init,
                    jac=True,
                    args=(state_mu, state_var),
                    method='L-BFGS-B',
                    bounds=self.actions_mapper.bounds,
                    options=self.config.controller.actions_optimizer_params)
                actions_mpc_restart = iter_res.x
                func_val = iter_res.fun
            else:
                actions_mpc_restart = generate_mpc_action_init_random(len_horizon=self.config.controller.len_horizon, dim_action=self.actions_mapper.dim_action)
                func_val, grad_val = self.compute_mean_lcb_trajectory(actions_mpc_restart, state_mu, state_var)
            
            if func_val < opt_fun or (actions_mpc_optim is None and np.isnan(func_val)):
                opt_fun = func_val
                actions_mpc_optim = actions_mpc_restart

        self.actions_mpc_previous_iter = actions_mpc_optim.copy()
        actions_mpc_optim = torch.Tensor(actions_mpc_optim)
        actions_model_optim = self.actions_mapper.transform_action_mpc_to_action_model(actions_mpc_optim)
        return actions_model_optim

    def _get_random_actions(self, state_mu:torch.Tensor, state_var:torch.Tensor) -> torch.Tensor:
        actions_mpc = generate_mpc_action_init_random(len_horizon=self.config.controller.len_horizon, dim_action=self.actions_mapper.dim_action)
        actions_model = self.actions_mapper.transform_action_mpc_to_action_model(torch.Tensor(actions_mpc))

        x_mem, y_mem = self.memory.get()
        with torch.no_grad():
            self.transition_model.prepare_inference(x_mem, y_mem)
        func_val, grad_val = self.compute_mean_lcb_trajectory(actions_mpc, state_mu, state_var) # store the trajectory info in the object
        return actions_model

    def add_memory(self, obs: np.array, action: np.array, obs_new:np.array, reward:float, predicted_state:np.array=None, predicted_state_std:np.array=None):
        """
        Add an observation, action and observation after applying the action to the memory that is used
        by the gaussian process models.
        At regular number of points interval (self.training_frequency),
        the training process of the gaussian process models will be launched to optimize the hyper-parameters.

        Args:
            obs: non-normalized observation. Dim=(Ns,)
            action: non-normalized action. Dim=(Ns,)
            obs_new: non-normalized observation obtained after applying the action on the observation.
                                    Dim=(Ns,)
            reward: reward obtained from the gym env. Unused at the moment.
                            The cost given state and action is computed instead.
            predicted_state:
                        if check_storage is True and predicted_state is not None,
                        the prediction error for that point will be computed.
                        and the point will only be stored in memory if the
                        prediction error is larger than self.error_pred_memory. Dim=(Ns,)

            predicted_state_std:
                        If check_storage is true, and predicted_state_std is not None, the point will only be
                        stored in memory if it is larger than self.error_pred_memory. Dim=(Ns,)

            where Ns: dimension of states, Na: dimension of actions
        """
        state_mu, state_var = self.observation_state_mapper.get_state(obs=obs, update_internals=False)
        state_mu_new, state_var_new = self.observation_state_mapper.get_state(obs=obs_new, update_internals=False)
        action_model = self.actions_mapper.transform_action_raw_to_action_model(action)

        self.memory.add(state_mu, action_model, state_mu_new, reward, iter_ctrl=self.iter_ctrl-1, predicted_state=predicted_state, predicted_state_std=predicted_state_std)

        if self.iter_ctrl % self.config.training.training_frequency == 0 and \
                not ('p_train' in self.__dict__ and not self.p_train._closed):
            self.start_training_process()

    def start_training_process(self):
        saved_state = self.transition_model.save_state()
        saved_state.to_arrays()
        self.p_train = self.ctx.Process(target=self.transition_model.train, 
                                        args=(
                                            self.queue_train,
                                            saved_state,
                                            self.config.training.lr_train, 
                                            self.config.training.iter_train, 
                                            self.config.training.clip_grad_value,
                                            self.config.training.print_train, 
                                            self.config.training.step_print_train)
                                        )
        self.p_train.start()

    def check_and_close_processes(self):
        """
        Check active parallel processes, wait for their resolution, get the parameters and close them
        """
        if 'p_train' in self.__dict__ and not self.p_train._closed and not (self.p_train.is_alive()):
            params_dict_list = self.queue_train.get()
            self.p_train.join()
            for model_idx in range(len(self.transition_model.models)):
                self.transition_model.models[model_idx].initialize(**params_dict_list[model_idx])
            self.p_train.close()
            x_mem, y_mem = self.memory.get()
            self.transition_model.prepare_inference(x_mem, y_mem)
    
    def compute_mean_lcb_trajectory(self, actions_mpc:np.array, obs_mu:torch.Tensor, obs_var:torch.Tensor):
        """
        Compute the mean lower bound cost of a trajectory given the actions of the trajectory
        and initial state distribution. The gaussian process models are used to predict the evolution of
        states (mean and variance). Then the cost is computed for each predicted state and the mean is returned.
        The partial derivatives of the mean lower bound cost with respect to the actions are also returned.
        They are computed automatically with autograd from pytorch.
        This function is called multiple times by an optimizer to find the optimal actions.

        Args:
            actions_mpc: actions to apply for the simulated trajectory.
                                    It is a flat 1d array so that this function can be used by the minimize function of the scipy library.
                                    It is reshaped and transformed into a tensor inside.
                                    If self.config.actions.limit_action_change is true, each element of the array contains the relative
                                    change with respect to the previous iteration, so that the change can be bounded by
                                    the optimizer. dim=(Nh x Na,)
                                    where Nh is the len of the horizon and Na the dimension of actions

            obs_mu:	mean value of the inital state distribution.
                                    dim=(Ns,) where Ns is the dimension of state

            obs_var: covariance matrix of the inital state distribution.
                                    dim=(Ns, Ns) where Ns is the dimension of state

        Returns:
            mean_cost_traj_lcb.item() (float): lower bound of the mean cost distribution
                                                        of the predicted trajectory.


            gradients_dcost_dactions.detach().numpy() (numpy.array):
                                                                Derivative of the lower bound of the mean cost
                                                                distribution with respect to each of the mpc actions in the
                                                                prediction horizon. Dim=(Nh * Na,)
                                                                where Nh is the len of the horizon and Ma the dimension of actions
        """
        # reshape actions from flat 1d numpy array into 2d tensor
        actions_mpc = torch.Tensor(actions_mpc)
        actions_mpc.requires_grad = True
        actions_model = self.actions_mapper.transform_action_mpc_to_action_model(action_mpc=actions_mpc)
        states_mu_pred, states_var_pred = self.transition_model.predict_trajectory(actions_model, obs_mu, obs_var, len_horizon=self.config.controller.len_horizon, current_time_idx=self.iter_ctrl)
        rewards_traj, rewards_traj_var = self.state_reward_mapper.get_rewards_trajectory(states_mu_pred, states_var_pred, actions_model)
        rewards_traj_ucb = rewards_traj + self.config.reward.exploration_factor * torch.sqrt(rewards_traj_var)

        if self.config.reward.clip_lower_bound_cost_to_0:
            clamp = self.clamp_lcb_class.apply
            rewards_traj_ucb = clamp(rewards_traj_ucb, float('-inf'), 0)
        mean_reward_traj_ucb = rewards_traj_ucb.mean()
        mean_cost_traj_ucb = -mean_reward_traj_ucb
        gradients_dcost_dactions = torch.autograd.grad(mean_cost_traj_ucb, actions_mpc, retain_graph=False)[0]

        self.cost_traj_mean_lcb = mean_reward_traj_ucb.detach()
        self.states_mu_pred = states_mu_pred.detach()
        self.rewards_trajectory = rewards_traj.detach()
        self.rewards_traj_var = rewards_traj_var.detach()
        self.states_var_pred = states_var_pred.detach()

        return mean_cost_traj_ucb.item(), gradients_dcost_dactions.detach().numpy()

    def compute_cost_unnormalized(self, obs:np.array, action:np.array, obs_var:np.array=None)-> "tuple[float]":
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
        state_mu, state_var = self.observation_state_mapper.get_state(obs=obs, obs_var=obs_var, update_internals=False)
        action_model = self.actions_mapper.transform_action_raw_to_action_model(action)
        reward_mu, reward_var = self.state_reward_mapper.get_reward(state_mu, state_var, action_model)
        return -reward_mu.item(), reward_var.item()

    def get_iter_info(self):
        return self.iter_info

    def store_iter_info(self, iter_info: IterationInformation):
        iter_info_dict = iter_info.__dict__
        for key in iter_info_dict.keys():
            if not key in self.info_iters:
                self.info_iters[key] = [iter_info_dict[key]]
            else:
                self.info_iters[key].append(iter_info_dict[key])

