import numpy as np
import gym
from gym import spaces


class ProcessControl(gym.Env):
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

	def __init__(self, dt=1, s_range=(9, 11), fi_range=(0., 0.2), ci_range=(0, 0.2),
						cr_range=(0.5, 1), noise_l_prop_range=(1e-5, 1e-3),
						noise_co_prop_range=(1e-5, 1e-3), sp_l_range=(0.2, 0.8),
						sp_co_range=(0.2, 0.4), change_params=True, period_change=50):
		"""
		The environment represents the simulation of a cuve with two input flow with different concentrations of product.
		The flow fi with concentration ci is not controllable
		The flow with concentration cr is controllable with action_1
		action_0 controls the output flow
		state0 = level
		state1 = concentration of product

		s: surface of the cuve
		ci: concentration of product with flow fi
		cr: concentration of product for input input_1

		There is a gaussian noise on the measurement of the level 
		noise_l_prop: noise on level
		noise_co: noise on concentration

		The parameters are taken randomly between the ranges provided.
		If change_params is set to True, the parameters are changed every period_change timesteps. 
		Can be used to see how the method is robust to environment changes
		dt: time between time steps
		"""
		super().__init__()
		self.name = 'processcontrol'
		# Definitions for gym env: action and observation space
		# They must be gym.spaces objects
		self.observation_space = spaces.Box(
			low=np.array([0, 0]),
			high=np.array([10, 1]),
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

	def define_params(self):
		"""
		Define the env parameters. It is not specific to gym
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

		if hasattr(self, 'v'):
			self.clip_parameters()

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

	def reset(self, min_prop=0.3, max_prop=0.7):
		# Reset the state of the environment to an initial state
		self.iter = 0
		ranges = (self.observation_space.high - self.observation_space.low)
		obs_sample = np.clip(self.observation_space.sample(),
							min_prop * ranges + self.observation_space.low,
							max_prop * ranges + self.observation_space.low)
		self.v = obs_sample[0] * self.s
		self.r = obs_sample[1] * self.v
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
		l_mes = np.clip(l_mes, self.observation_space.low[0], self.observation_space.high[0])
		co_mes = np.clip(co_mes, self.observation_space.low[1], self.observation_space.high[1])
		return np.array([l_mes, co_mes])

	def render(self, mode='human', close=False):
		# Render the environment
		pass

	def clip_parameters(self, prop_level_max_after_reset=0.9):
		v_p = self.v
		self.v = np.clip(self.v, 
							a_min=0, 
							a_max=prop_level_max_after_reset * (self.s * self.observation_space.high[0])
						)
		self.r = self.r * self.v / v_p 
