from rl_gp_mpc.config_classes.total_config import Config


class BaseControllerObject:
    def __init__(self, config: Config):
        raise NotImplementedError

    def add_memory(self, obs, action, obs_new, reward, **kwargs):
        raise NotImplementedError()

    def get_action(self, obs_mu, obs_var=None):
        raise NotImplementedError()

    def get_action_random(self, obs_mu, obs_var=None):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()