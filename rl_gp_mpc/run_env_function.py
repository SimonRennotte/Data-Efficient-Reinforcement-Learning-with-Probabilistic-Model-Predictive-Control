import time
import numpy as np
import matplotlib.pyplot as plt
from gym.core import Env

from rl_gp_mpc import GpMpcController
from rl_gp_mpc import ControlVisualizations
from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.config_classes.total_config import Config

NUM_DECIMALS_REPR = 3
np.set_printoptions(precision=NUM_DECIMALS_REPR, suppress=True)

def run_env(env: Env, control_config:Config, visu_config: VisuConfig, random_actions_init=10, num_steps=150, verbose=True):
    visu_obj = ControlVisualizations(env=env, num_steps=num_steps, control_config=control_config, visu_config=visu_config)

    ctrl_obj = GpMpcController(observation_low=env.observation_space.low,
                                observation_high=env.observation_space.high, 
                                action_low=env.action_space.low,
                                action_high=env.action_space.high, 
                                config=control_config)

    obs = env.reset()
    
    for idx_ctrl in range(num_steps):
        action_is_random = (idx_ctrl < random_actions_init)
        action = ctrl_obj.get_action(obs_mu=obs, random=action_is_random)

        iter_info = ctrl_obj.get_iter_info()

        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
        visu_obj.update(obs=obs, reward=-cost, action=action, env=env, iter_info=iter_info)

        obs_new, reward, done, info = env.step(action)

        ctrl_obj.add_memory(obs=obs, action=action, obs_new=obs_new,
                            reward=-cost,
                            predicted_state=iter_info.predicted_states[1],
                            predicted_state_std=iter_info.predicted_states_std[1])
        obs = obs_new
        if verbose:
            print(str(iter_info))


    visu_obj.save(ctrl_obj)
    ctrl_obj.check_and_close_processes()
    env.__exit__()
    visu_obj.close() 
    return visu_obj.get_costs()


def run_env_multiple(env, env_name, control_config:Config, visu_config: VisuConfig, num_runs, random_actions_init=10, num_steps=150, verbose=True):
    costs_runs = []
    for run_idx in range(num_runs):
        costs_iter = run_env(env, control_config, visu_config, random_actions_init, num_steps, verbose=verbose)
        costs_runs.append(costs_iter)
        time.sleep(1)

    costs_runs = np.array(costs_runs)

    costs_runs_mean = np.mean(costs_runs, axis=0)
    costs_runs_std = np.std(costs_runs, axis=0)

    x_axis = np.arange(len(costs_runs_mean))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, costs_runs_mean)
    ax.fill_between(x_axis, costs_runs_mean - costs_runs_std, costs_runs_mean + costs_runs_std, alpha=0.4)
    plt.title(f"Costs of multiples {env_name} runs")
    plt.ylabel("Cost")
    plt.xlabel("Env iteration")
    plt.savefig(f'multiple_runs_costs_{env_name}.png')
    plt.show()