
class AbstractEmbedActionsMapper:
	def __init__(self, action_space, config_actions):
		self.config = config_actions
		self.action_low = action_space.low
		self.action_high = action_space.high
	
	def embed_actions_to_actions(self, embed_actions):
		raise NotImplementedError

	def generate(self):
		raise NotImplementedError
	
	def train(self, actions_history):
		raise NotImplementedError


class NormedActionsMapper:
	def __init__(self, action_space, config_actions):
		super().__init__(action_space, config_actions)
	
	def embed_actions_to_actions(self, embed_actions):
		pass

	def train(self, actions_history):
		raise NotImplementedError()


class DerivativeActionsMapper:
	def __init__(self, action_space, config_actions):
		super().__init__(action_space, config_actions)
	
	def embed_actions_to_actions(self, embed_actions):
		pass

	def train(self, actions_history):
		raise NotImplementedError()

