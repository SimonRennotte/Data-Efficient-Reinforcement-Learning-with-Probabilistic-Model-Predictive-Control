import os
import multiprocessing

import numpy as np
import matplotlib
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.config_classes.total_config import Config
from rl_gp_mpc.control_objects.controllers.iteration_info_class import IterationInformation
from rl_gp_mpc.control_objects.controllers.gp_mpc_controller import GpMpcController

from .utils import get_env_name, create_folder_save
from .static_3d_graph import save_plot_model_3d_process
from .static_2d_graph import save_plot_2d
from .dynamic_2d_graph import LivePlotParallel, LivePlotSequential


matplotlib.rc('font', size='6')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
ALPHA_CONFIDENCE_BOUNDS = 0.3
FONTSIZE = 6

# Values used for graph visualizations only
MEAN_PRED_COST_INIT = 1
STD_MEAN_PRED_COST_INIT = 10


class ControlVisualizations:
    def __init__(self, env, num_steps:int, control_config: Config, visu_config: VisuConfig, num_repeat_actions:int):
        self.control_config = control_config
        self.visu_config = visu_config
        self.num_repeat_actions = num_repeat_actions

        self.observations = []
        self.actions = []
        self.costs = []
        self.iter_infos = []

        self.dynamic_graph = None
        self.gym_animation_graph = None

        self.plots_2d_save_process = None
        self.env = env

        self.rec = None

        self.env_str = get_env_name(env)
        self.folder_save = create_folder_save(self.env_str)

        if self.visu_config.render_live_plot_2d:
            self.init_dynamic_graph(env, num_steps)

        if self.visu_config.save_render_env:
            self.init_env_visu(self.env_str)

        self.ctx = multiprocessing.get_context('spawn')

    def update(self, obs:np.ndarray, action:np.ndarray, cost:float, iter_info:IterationInformation=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.costs.append(cost)
        self.iter_infos.append(iter_info)
        self.live_plot_obj.update(obs=obs, action=action, cost=cost, iter_info=iter_info)
        self.env_render_step()

    def save(self, random_actions_init):
        self.save_plots_2d(states=np.array(self.observations), actions=np.array(self.actions), rewards=-np.array(self.costs), random_actions_init=random_actions_init, info_iters=None)
        self.save_plots_model_3d()

    def close(self):
        if self.rec is not None:
            self.rec.close()
        if self.live_plot_obj is not None:
            self.live_plot_obj.graph_p.terminate()

        #self.close_running_processes()
        #self.save()
        #self.close_running_processes()

    def init_dynamic_graph(self, env, num_steps):
        if self.visu_config.run_live_graph_parallel_process:
            self.live_plot_obj = LivePlotParallel(num_steps,
                env.observation_space, env.action_space,
                step_pred=self.num_repeat_actions,
                use_constraints=bool(self.control_config.reward.use_constraints),
                state_min=self.control_config.reward.state_min,
                state_max=self.control_config.reward.state_max)
        else:
            self.live_plot_obj = LivePlotSequential(num_steps,
                env.observation_space, env.action_space,
                step_pred=self.num_repeat_actions,
                use_constraints=bool(self.control_config.reward.use_constraints),
                state_min=self.control_config.reward.state_min,
                state_max=self.control_config.reward.state_max)

    def init_env_visu(self, env):
        try:
            self.rec = VideoRecorder(env, path=os.path.join(self.folder_save, 'anim' + self.env_str + '.mp4'))
        except:
            pass

    def env_render_step(self):
        if self.visu_config.render_env:
            try: self.env.render()
            except: pass
        if self.rec is not None:
            try: self.rec.capture_frame()
            except: pass

    def save_plots_model_3d(self, controller: GpMpcController):
        """
        Launch the process in which the 3d plots are computed and saved.
        It contains points in memory, predictions of each gaussian process (1 by dimension of state).
        Args:
            prop_extend_domain (float): if set to 1, the 3d models will show the domain covered by the points in memory.
                                        If it is larger, the domain will be larger, if is less than 1 it will be lower.
                                        Default=1
            n_ticks (int): number of points to predict with the gaussian processes
                            in each axis (n_ticks * n_ticks points in total). Default=100
            total_col_max (int): maximum number of columns to show on one window containing
                                    multiple gaussian proces models. Default=3

        """
        if not ('p_save_plots_model_3d' in self.__dict__ and not self.p_save_plots_model_3d._closed):
            self.p_save_plots_model_3d = self.ctx.Process(target=save_plot_model_3d_process,
                args=(controller.x[:controller.len_mem], controller.y[:controller.len_mem],
                [model.state_dict() for model in controller.models],
                controller.gp_constraints, self.folder_save, [controller.idxs_mem_gp] * controller.num_states))
            self.p_save_plots_model_3d.start()

    def save_plots_2d(self, states, actions, rewards, info_iters, random_actions_init):
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
            #states = self.to_normed_obs_tensor(np.array(states))
            #actions = self.to_normed_action_tensor(np.array(actions))
            costs = - np.array(rewards)
            self.p_save_plots_2d = self.ctx.Process(target=save_plot_2d,
                args=(states, actions, costs, info_iters, self.folder_save,
                        self.control_config.num_repeat_actions, 
                        random_actions_init, self.control_config.state_min, self.control_config.state_max))
            self.p_save_plots_2d.start()

    def close_running_processes(self):
        if 'p_save_plots_2d' in self.__dict__ \
                and not self.p_save_plots_2d._closed and not (self.p_save_plots_2d.is_alive()):
            self.p_save_plots_2d.join()
            self.p_save_plots_2d.close()

        if 'p_save_plots_model_3d' in self.__dict__ \
                and not self.p_save_plots_model_3d._closed and not (self.p_save_plots_model_3d.is_alive()):
            self.p_save_plots_model_3d.join()
            self.p_save_plots_model_3d.close()
