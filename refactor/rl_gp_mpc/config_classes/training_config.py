class TrainingConfig:
	def __init__(
		self,
		lr_train=7e-3,
		iter_train=15 ,
		training_frequency=10,
		clip_grad_value=1e-3,
		print_train=False,
		step_print_train=5
	):
		self.lr_train = lr_train
		self.iter_train = iter_train
		self.training_frequency = training_frequency
		self.clip_grad_value = clip_grad_value
		self.print_train = print_train
		self.step_print_train = step_print_train