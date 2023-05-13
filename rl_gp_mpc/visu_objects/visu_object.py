import os
import multiprocessing
import copy
import time

import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.core import Env

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.config_classes.total_config import Config
from rl_gp_mpc.control_objects.controllers.iteration_info_class import IterationInformation
from rl_gp_mpc.control_objects.controllers.gp_mpc_controller import GpMpcController

from .utils import get_env_name, create_folder_save
from .static_3d_graph import save_plot_model_3d
from .static_2d_graph import save_plot_2d
from .dynamic_2d_graph import LivePlotParallel


class ControlVisualizations:
    def __init__(self, env:Env, num_steps:int, control_config: Config, visu_config: VisuConfig):
        self.control_config = control_config
        self.visu_config = visu_config

        self.states = []
        self.actions = []
        self.rewards = []
        self.model_iter_infos = []

        self.rec = None
        self.use_thread = False

        self.env_str = get_env_name(env)
        self.folder_save = create_folder_save(self.env_str)

        self.obs_min = env.observation_space.low
        self.obs_max = env.observation_space.high
        self.action_min = env.action_space.low
        self.action_max = env.action_space.high

        self.init = True

        if self.visu_config.render_live_plot_2d:
            self.live_plot_obj = LivePlotParallel(num_steps,
                use_constraints=bool(self.control_config.reward.use_constraints),
                dim_states=len(self.obs_min),
                dim_actions=len(self.action_min),
                state_min=self.control_config.reward.state_min,
                state_max=self.control_config.reward.state_max,
                use_thread=self.use_thread,
                path_save=os.path.join(self.folder_save, 'control_animation.mp4'),
                save=self.visu_config.save_live_plot_2d)

        if self.visu_config.save_render_env:
            self.rec = VideoRecorder(env, path=os.path.join(self.folder_save, 'gym_animation.mp4'))
        
        self.processes_running = True

    def update(self, obs:np.ndarray, action:np.ndarray, reward:float, env:Env, iter_info:IterationInformation=None):
        state = (obs - self.obs_min) / (self.obs_max - self.obs_min)
        action_norm = (action - self.action_min) / (self.action_max - self.action_min)
        self.states.append(state)
        self.actions.append(action_norm)
        self.rewards.append(reward)

        iter_info_arrays = copy.deepcopy(iter_info)
        iter_info_arrays.to_arrays()
        self.model_iter_infos.append(iter_info_arrays)

        if self.visu_config.render_live_plot_2d:
            self.live_plot_obj.update(state=state, action=action_norm, cost=-reward, iter_info=iter_info_arrays)
        self.env_render_step(env)
        if self.init:
            time.sleep(2)
            self.init = False

    def env_render_step(self, env):
        if self.visu_config.render_env:
            env.render()
        if self.visu_config.save_render_env:
            self.rec.capture_frame()

    def save_plot_model_3d(self, controller: GpMpcController):
        save_plot_model_3d(controller.transition_model.save_state(), self.folder_save, controller.memory)

    def save_plot_2d(self):
        save_plot_2d(np.array(self.states), np.array(self.actions), -np.array(self.rewards), 
                        self.model_iter_infos, self.folder_save, 
                        self.control_config.reward.use_constraints,
                        self.control_config.reward.state_min, 
                        self.control_config.reward.state_max)

    def save(self, controller: GpMpcController):
        self.save_plot_2d()
        self.save_plot_model_3d(controller)

    def close(self):
        if self.visu_config.save_render_env:
            self.rec.close()
        self.close_running_processes()

    def close_running_processes(self):
        if 'live_plot_obj' in self.__dict__ and self.live_plot_obj.graph_p.is_alive():
            self.live_plot_obj.close()
        
        self.processes_running = False

    def __exit__(self):
        if self.processes_running:
            self.close()

    def get_costs(self):
        return -np.array(self.rewards)
