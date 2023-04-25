class VisuConfig:
	def __init__(
		self,
		save_render_env=True,
		render_live_plot_2d=True,
		run_live_graph_parallel_process=True,
		render_env=True,
		freq_iter_save_plots=50
	):
		self.save_render_env = save_render_env
		self.render_live_plot_2d = render_live_plot_2d
		self.run_live_graph_parallel_process = run_live_graph_parallel_process
		self.render_env = render_env
		self.freq_iter_save_plots = freq_iter_save_plots