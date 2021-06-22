import numpy as np
import gym
from gym import spaces


class CustomEnv(gym.Env):
	"""
	Example of custom Environment that follows gym interface.
	The additional methods:
		- define_params: used to be able to vary the internal parameters
				of the env to check the robustness of the control to change in the env.
		- get_obs: methods that returns the observation given the state of the env.
					It is used to add noise to measurement.
					For example, the state is the volume, but the level is measured
		If change_params is set to True, the internal params will be changed every period_change iterations with the env
		Params are ranges because they are chosen at random between the two values provided. """
	metadata = {'render.modes': []}

	def __init__(self, dt=1, s_range=(10, 20), fi_range=(0.2, 0.4), ci_range=(0, 0.2),
						cr_range=(0.4, 0.9), noise_l_prop_range=(1e-5, 3e-3),
						noise_co_prop_range=(1e-5, 3e-3), sp_l_range=(0.2, 0.8),
						sp_co_range=(0.2, 0.4), change_params=False, period_change=50):
		super(CustomEnv, self).__init__()
		# Definitions for gym env: action and observation space
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

		# definitions specific to my env
		self.dt = dt
		self.s_range = s_range
		self.fi_range = fi_range
		self.ci_range = ci_range
		self.cr_range = cr_range
		self.noise_l_prop_range = noise_l_prop_range
		self.noise_co_prop_range = noise_co_prop_range
		self.sp_l_range = sp_l_range
		self.sp_co_range = sp_co_range
		self.change_params = change_params
		self.period_change = period_change
		# initialization of parameters and states
		self.define_params()
		_ = self.reset()

	def define_params(self):
		"""
		Define my env parameters. It is not specific to gym
		"""
		self.s = np.random.uniform(self.s_range[0], self.s_range[1])
		self.fi = np.random.uniform(self.fi_range[0], self.fi_range[1])
		self.ci = np.random.uniform(self.ci_range[0], self.ci_range[1])
		self.cr = np.random.uniform(self.cr_range[0], self.cr_range[1])
		self.noise_l_prop = np.exp(np.random.uniform(
									np.log(self.noise_l_prop_range[0]),
									np.log(self.noise_l_prop_range[1])))
		self.noise_co_prop = np.exp(np.random.uniform(
									np.log(self.noise_co_prop_range[0]),
									np.log(self.noise_co_prop_range[1])))
		self.sp_l = np.random.uniform(self.sp_l_range[0], self.sp_l_range[1])
		self.sp_co = np.random.uniform(self.sp_co_range[0], self.sp_co_range[1])
		print("New params value: s: {:.2f},  fi: {:.2f},  ci: {:.2f}, cr: {:.2f}, "
			"noise_l: {:.4f}, noise_co: {:.4f}, sp_l: {:.2f}, sp_co: {:.2f}".format(
			  self.s, self.fi, self.ci, self.cr, self.noise_l_prop, self.noise_co_prop, self.sp_l, self.sp_co))

	def step(self, action):
		dv = (self.fi + action[1] - action[0])
		dr = (self.fi * self.ci + action[1] * self.cr - action[0] * self.r / (self.v + 1e-3))
		self.v += dv * self.dt
		self.r += dr * self.dt
		self.iter += 1

		self.v = np.clip(self.v,
						self.observation_space.low[0] * self.s,
						self.observation_space.high[0] * self.s)
		self.r = np.clip(self.r,
						self.observation_space.low[1] * self.v,
						self.observation_space.high[1] * self.v)

		reward = - (pow(self.v / self.s - self.sp_l, 2) + pow(self.r / (self.v + 1e-6) - self.sp_co, 2))
		done = 0
		info = {}
		if self.change_params and self.iter % self.period_change == 0:
			self.define_params()

		return self.get_obs(), reward, done, info

	def reset(self, min_prop=0.1, max_prop=0.9):
		# Reset the state of the environment to an initial state
		self.iter = 0
		ranges = (self.observation_space.high - self.observation_space.low)
		obs_sample = np.clip(self.observation_space.sample(),
							min_prop * ranges + self.observation_space.low,
							max_prop * ranges + self.observation_space.low)
		self.v = obs_sample[0] * self.s
		self.r = 0 * self.v
		return self.get_obs()

	def get_obs(self):
		"""
		Get observation given the internal states. It is not specific to gym
		"""
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
