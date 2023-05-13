class VisuConfig:
	def __init__(
		self,
		save_render_env:bool=True,
		render_live_plot_2d:bool=True,
		render_env:bool=True,
		save_live_plot_2d:bool=False
	):
		"""
		save_render_env: if set to true and render_env is true, will save the env animation
		render_live_plot_2d: if true, will show the dynamic 2d graph update in real time
		render_env: if set to true, will show the env
		save_live_plot_2d: if set to true and render_live_plot_2d is true, will save the 2d graph animation

		"""
		self.save_render_env = save_render_env
		self.render_live_plot_2d = render_live_plot_2d
		self.render_env = render_env
		self.save_live_plot_2d = save_live_plot_2d