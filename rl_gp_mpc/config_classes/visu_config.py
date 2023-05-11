class VisuConfig:
	def __init__(
		self,
		save_render_env:bool=True,
		render_live_plot_2d:bool=True,
		render_env:bool=True,
		freq_iter_save_plots:int=50
	):
		self.save_render_env = save_render_env
		self.render_live_plot_2d = render_live_plot_2d
		self.render_env = render_env
		self.freq_iter_save_plots = freq_iter_save_plots