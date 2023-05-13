class VisuConfig:
	def __init__(
		self,
		save_render_env:bool=True,
		render_live_plot_2d:bool=True,
		render_env:bool=True,
		save_live_plot_2d:bool=False
	):
		self.save_render_env = save_render_env
		self.render_live_plot_2d = render_live_plot_2d
		self.render_env = render_env
		self.save_live_plot_2d = save_live_plot_2d