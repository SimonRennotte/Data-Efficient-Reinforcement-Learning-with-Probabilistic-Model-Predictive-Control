import torch


class Memory:
    def __init__(self, config_memory):
        self.config = config_memory

    def add(self, observation, action, observation_next, reward=None):
        pass

    def update(self):
        pass


