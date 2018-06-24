import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from time import time
from util.game_info import as_to_str
from util.stats import DistributionInfo as Stat


class RunInfo:
	def __init__(self):
		self.net = None
		self.replay_memory = None
		self.run_start_time = time()

		self.iteration_count = 0
		self.last_iter_time = time()
		self.action_stat = {}
		self.user_action_count = 0

		self.episode_count = 0
		self.last_ep_time = time()
		self.last_ep_iter = 0
		self.episode_lengths = []
		self.episode_times = []
		self.mem_up_times = []
		self.train_times = []

		self.state_score_data = {"angle_to_ball": 0, "dist_from_ball": 0, "speed": 0, "boost": 0, "super_sonic": 0}

	def episode(self, mem_up_time, train_time, verbose=True):
		ep_end_time = time()
		self.episode_lengths.append(self.iteration_count-self.last_ep_iter)
		self.episode_times.append(ep_end_time-self.last_ep_time)
		self.mem_up_times.append(mem_up_time)
		self.train_times.append(train_time)
		self.episode_count += 1

		if verbose:
			print("\nEpisode {0:d} over".format(self.episode_count - 1))
			print("Total episode time:", self.episode_times[-1])
			print("Iterations this episode:", self.episode_lengths[-1])
			print("Memory update took {0:.2f}sec".format(mem_up_time))
			print("training took {0:.2f}sec".format(train_time))
			print("Action stat:", self.action_stat)
			print("User Actions:", self.user_action_count)
			print("Average estimation error:", Stat(self.replay_memory.estimation_errors))

		self.write_data()

	def iteration(self, action, user=False, verbose=False):
		if user:
			self.user_action_count += 1
		try:
			self.action_stat[as_to_str(action)] += 1
		except KeyError:
			self.action_stat[as_to_str(action)] = 1

		self.iteration_count += 1

		if verbose:
			print("Iteration:", self.iteration_count)
			print("Actor:", ("user" if user else "agent"))
			print("Action:", action)

	def state_score(self, angle_to_ball, dist_from_ball, speed, boost, super_sonic):
		self.state_score_data["angle_to_ball"] += angle_to_ball
		self.state_score_data["dist_from_ball"] += dist_from_ball
		self.state_score_data["speed"] += speed
		self.state_score_data["boost"] += boost
		self.state_score_data["super_sonic"] += super_sonic

	# todo split into iteration and episode data
	def write_data(self):
		print(os.path.abspath("./"))
		np.savetxt("E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/temp/estimation_errors.csv", self.replay_memory.estimation_errors, delimiter=",")


def update_graphs(i):
	for info in infos:
		vals = np.genfromtxt("./temp/" + info["file"], delimiter=",")
		info["plot_func"](info["axis"], vals)


def est_errs_plot(axis, vals):
	bins = [i for i in range(25)]
	axis.clear()
	axis.hist(vals, bins, histtype="bar")


if __name__ == "__main__":
	style.use("fivethirtyeight")
	figure = plt.figure()

	infos = [
		{"title": "Estimation Errors", "file": "estimation_errors.csv", "axis": figure.add_subplot(1, 1, 1), "plot_func": est_errs_plot}
	]

	ani = animation.FuncAnimation(figure, update_graphs, interval=1000)
	plt.show()
