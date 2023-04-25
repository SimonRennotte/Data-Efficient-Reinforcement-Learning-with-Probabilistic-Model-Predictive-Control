import time

import gym
import torch

from rl_gp_mpc import GpMpcController
from rl_gp_mpc import ControlVisualizations
from rl_gp_mpc.config_classes.visu_config import VisuConfig
from configs.config_pendulum import get_config

random_actions_init = 25
num_steps = 150
num_repeat_actions = 1
verbose = True
render_env = True
save_plots_2d = False
save_plots_model_3d = False
freq_iter_save_plots = 25

def main():
	# choose the configuration file to load the corresponding env
	# open(os.path.join('params', 'main_parameters_mountain_car.json'))
	# open(os.path.join('params', 'main_parameters_my_env.json'))
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    control_config = get_config(num_repeat_actions=num_repeat_actions, len_horizon=10)
    visu_config = VisuConfig()

    visu_obj = ControlVisualizations(env=env, num_steps=num_steps, control_config=control_config, visu_config=visu_config)
    ctrl_obj = GpMpcController(observation_space=env.observation_space, action_space=env.action_space, config=control_config)

    obs = env.reset()
    
    for idx_init in range(random_actions_init // num_repeat_actions):
        action = env.action_space.sample()
        for idx_action_repeat in range(num_repeat_actions):
            obs_new, reward, done, info = env.step(action)
            cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs_new, action)
            visu_obj.update(obs=obs_new, cost=cost, action=action)
            
        ctrl_obj.add_memory(obs=obs, action=action, obs_new=obs_new,
                            reward=-cost, check_storage=False)
        obs = obs_new
        # Necessary to store the last action in case that the parameter limit_action_change is set to True
        ctrl_obj.action_previous_iter = torch.Tensor(action)

    for idx_ctrl in range(random_actions_init // num_repeat_actions, num_steps//num_repeat_actions):
        time_start = time.time()
        action = ctrl_obj.compute_action(obs_mu=obs)
        iter_info = ctrl_obj.get_iter_info()

        for idx_action_repeat in range(num_repeat_actions):
            obs_new, reward, done, info = env.step(action)
            cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
            visu_obj.update(obs=obs_new, cost=cost, action=action, iter_info=iter_info)

        ctrl_obj.add_memory(obs=obs, action=action, obs_new=obs_new,
                            reward=-cost, check_storage=True,
                            predicted_state=iter_info.predicted_states[0],
                            predicted_state_std=iter_info.predicted_states_std[0])
        if verbose:
            iter_info_dict = iter_info.__dict__
            for key in iter_info_dict:
                print(key + ': ' + str(iter_info_dict[key]))

        obs = obs_new
        print('time loop: ' + str(time.time() - time_start) + ' s\n')

    visu_obj.save(random_actions_init=random_actions_init)
    env.__exit__()
    ctrl_obj.check_and_close_processes()
    visu_obj.close() 


if __name__ == '__main__':
	main()
