import datetime
import os

import matplotlib.pyplot as plt
import numpy as np


def get_env_name(env):
    try:
        env_name = env.env.spec.entry_point.replace('-', '_').replace(':', '_').replace('.', '_')
    except:
        env_name = env.name
    return env_name


def create_folder_save(env_name):
	datetime_now = datetime.datetime.now()
		
	folder_save = os.path.join('folder_save', env_name, datetime_now.strftime("%Y_%m_%d_%H_%M_%S"))
	if not os.path.exists(folder_save):
		os.makedirs(folder_save)
	return folder_save


def close_run(ctrl_obj, env, obs_lst, actions_lst, rewards_lst, random_actions_init,
					live_plot_obj=None, rec=None, save_plots_3d=False, save_plots_2d=False):
	"""
	Close all visualisations and parallel processes that are still running.
	Save all visualisations one last time if save args set to True
	Args:
		ctrl_obj:
		env (gym env): gym environment
		obs_lst (list of numpy array): Contains all the past observations
		actions_lst (list of numpy array): Contains all the past actions
		rewards_lst(list): Contains all the past rewards (or -cost)
		random_actions_init (int): number of initial actions where the controller was not used (random actions)
		live_plot_obj (object or None): object used for visualisation in real time of the control in a 2d graph.
								If None, not need to close it. Default=None
		rec (object or None): object used to visualise the gym env in real-time.
								If None, no need to close it. default=None
		save_plots_3d (bool): If set to True, will save the 3d plots. Default=False
		save_plots_2d (bool): If set to True, will save the 2d plots. Default=False
	"""

	env.__exit__()
	if rec is not None:
		rec.close()
	if live_plot_obj is not None:
		live_plot_obj.graph_p.terminate()
	# save plots at the end
	ctrl_obj.check_and_close_processes()
	if save_plots_3d:
		ctrl_obj.save_plots_model_3d()
		# wait for the process to be finished
		ctrl_obj.p_save_plots_model_3d.join()
		ctrl_obj.p_save_plots_model_3d.close()
	if save_plots_2d:
		ctrl_obj.save_plots_2d(states=obs_lst, actions=actions_lst, rewards=rewards_lst,
								random_actions_init=random_actions_init)
		# wait for the process to be finished
		ctrl_obj.p_save_plots_2d.join()
		ctrl_obj.p_save_plots_2d.close()
	plt.close()


def save_plots_2d(self, states, actions, rewards, random_actions_init):
    """
    Launch the parallel process in which the 2d plots are computed and saved.
    Args:
        states (list of numpy.array): past observations
        actions (list of numpy.array): past actions
        rewards (list): past reward (or -cost)
        random_actions_init (int): initial number of actions preformed at random,
                                    without using the models to find the optimal actions
    """
    if not ('p_save_plots_2d' in self.__dict__ and not self.p_save_plots_2d._closed):
        states = self.to_normed_obs_tensor(np.array(states))
        actions = self.to_normed_action_tensor(np.array(actions))
        costs = - np.array(rewards)
        self.p_save_plots_2d = self.ctx.Process(target=save_plot_2d,
            args=(states, actions, costs, self.info_iters, self.folder_save,
                    self.config.controller.num_repeat_actions, random_actions_init, self.config.controller_constraints.state_min,self.config.controller_constraints.state_max))
        self.p_save_plots_2d.start()
        self.num_cores_main -= 1
