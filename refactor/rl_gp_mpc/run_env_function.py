import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from gym.core import Env

from rl_gp_mpc import GpMpcController
from rl_gp_mpc import ControlVisualizations
from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.config_classes.total_config import Config


def run_env(env: Env, control_config:Config, visu_config: VisuConfig, random_actions_init=10, num_repeat_actions=1, num_steps=150, verbose=True):
    visu_obj = ControlVisualizations(env=env, num_steps=num_steps, control_config=control_config, visu_config=visu_config, num_repeat_actions=num_repeat_actions)

    ctrl_obj = GpMpcController(observation_low=env.observation_space.low,
                                observation_high=env.observation_space.high, 
                                action_low=env.action_space.low,
                                action_high=env.action_space.high, 
                                config=control_config)

    obs = env.reset()
    
    for idx_ctrl in range(num_steps//num_repeat_actions):
        action_is_random = (idx_ctrl < (random_actions_init//num_repeat_actions))
        action = ctrl_obj.get_action(obs_mu=obs, random=action_is_random)
        iter_info = ctrl_obj.get_iter_info()

        for idx_action_repeat in range(num_repeat_actions):
            obs_new, reward, done, info = env.step(action)
            cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs_new, action)
            visu_obj.update(obs=obs_new, cost=cost, action=action, iter_info=iter_info)

        ctrl_obj.add_memory(obs=obs, action=action, obs_new=obs_new,
                            reward=-cost,
                            predicted_state=iter_info.predicted_states[0],
                            predicted_state_std=iter_info.predicted_states_std[0])
        obs = obs_new
        if verbose:
            print(str(iter_info))


    #visu_obj.save(random_actions_init=random_actions_init)
    env.__exit__()
    ctrl_obj.check_and_close_processes()
    visu_obj.close() 
    return visu_obj.costs


def run_env_multiple(env, control_config:Config, visu_config: VisuConfig, num_runs, random_actions_init=10, num_repeat_actions=1, num_steps=150, verbose=True):
    costs_runs = []
    for run_idx in range(num_runs):
        costs_iter = run_env(env, control_config, visu_config, random_actions_init, num_repeat_actions, num_steps, verbose=verbose)
        costs_runs.append(costs_iter)

    costs_runs = np.array(costs_runs)

    costs_runs_mean = np.mean(costs_runs, axis=0)
    costs_runs_std = np.std(costs_runs, axis=0)

    x_axis = np.arange(len(costs_runs_mean))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, costs_runs_mean)
    ax.fill_between(x_axis, costs_runs_mean - costs_runs_std, costs_runs_mean + costs_runs_std, alpha=0.4)
    plt.title("Costs of multiples mountain car runs")
    plt.ylabel("Cost")
    plt.xlabel("Env iteration")
    plt.show()