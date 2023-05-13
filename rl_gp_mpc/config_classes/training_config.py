class TrainingConfig:
	def __init__(
		self,
		lr_train:float=7e-3,
		iter_train:int=15,
		training_frequency:int=25,
		clip_grad_value:float=1e-3,
		print_train:bool=False,
		step_print_train:int=5
	):
		"""
		lr_train: learning rate when training the hyperparameters of the model
		iter_train: number of iteration when training the model
		training_frequency: the model will be trained periodically. The interval in number of control iteration is this parameter
		clip_grad_value: if the gradient is above this number, it will be clipped to that number. Allows for better stability
		print_train: if set to true, will print the training loop information
		step_print_train: if print_train is true, will print the info every step_print_train of the training iterations
		"""
		self.lr_train = lr_train
		self.iter_train = iter_train
		self.training_frequency = training_frequency
		self.clip_grad_value = clip_grad_value
		self.print_train = print_train
		self.step_print_train = step_print_train