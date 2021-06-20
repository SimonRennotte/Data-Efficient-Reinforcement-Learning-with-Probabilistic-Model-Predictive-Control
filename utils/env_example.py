import numpy as np
import gym
from gym import spaces


class CustomEnv(gym.Env):
	"""Example of custom Environment that follows gym interface"""
	metadata = {'render.modes': []}

	def __init__(self, s=10, dt=1, fi=0.33, ci=0.1, cr=0.34,
						noise_l_prop=3e-3, noise_co_prop=3e-3, sp_l=0.5, sp_co=0.5):
		super(CustomEnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		self.observation_space = spaces.Box(
			low=np.array([0, 0]),
			high=np.array([5, 1]),
			shape=(2,), dtype=np.float32)
		self.reward_range = (0, 1)
		self.action_space = spaces.Box(
			low=np.array([0, 0]),
			high=np.array([1, 1]),
			shape=(2,), dtype=np.float32)

		self.s = s
		self.dt = dt
		self.fi = fi
		self.ci = ci
		self.cr = cr
		self.noise_l_prop = noise_l_prop
		self.noise_co_prop = noise_co_prop
		self.sp_l = sp_l
		self.sp_co = sp_co

		_ = self.reset()

	def step(self, action):
		dv = (self.fi + action[1] - action[0])
		dr = (self.fi * self.ci + action[1] * self.cr - action[0] * self.r / (self.v + 1e-3))
		self.v += dv * self.dt
		self.r += dr * self.dt
		self.t += self.dt

		self.v = np.clip(self.v,
						self.observation_space.low[0] * self.s,
						self.observation_space.high[0] * self.s)
		self.r = np.clip(self.r,
						self.observation_space.low[1] * self.v,
						self.observation_space.high[1] * self.v)

		reward = - (pow(self.v / self.s - self.sp_l, 2) + pow(self.r / (self.v + 1e-6) - self.sp_co, 2))
		done = 0
		info = {}
		return self.get_obs(), reward, done, info

	def reset(self, min_prop=0.1, max_prop=0.9):
		# Reset the state of the environment to an initial state
		self.t = 0
		ranges = (self.observation_space.high - self.observation_space.low)
		obs_sample = np.clip(self.observation_space.sample(),
							min_prop * ranges + self.observation_space.low,
							max_prop * ranges + self.observation_space.low)
		self.v = obs_sample[0] * self.s
		self.r = obs_sample[1] * self.v
		return self.get_obs()

	def get_obs(self):
		l_mes = self.v / self.s
		co_mes = self.r / (self.v + 1e-6)
		if self.noise_l_prop != 0:
			l_mes += np.random.normal(0, self.noise_l_prop * self.observation_space.high[0])
		if self.noise_co_prop != 0:
			co_mes += np.random.normal(0, self.noise_co_prop * self.observation_space.high[1])
		return np.array([l_mes, co_mes])

	def render(self, mode='human', close=False):
		# Render the environment to the screen
		pass
