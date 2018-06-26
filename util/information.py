import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import util.stats as stats
from matplotlib import style
from time import time
from util.game_info import as_to_str
from util.stats import DistributionInfo as Stat

TEMP_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/temp/"
LOG_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/logs/"
ENTRY_SEPARATOR = "\n::Next Entry::\n"


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

		self.last_ep_iter = self.iteration_count
		self.last_ep_time = time()

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

	def write(self):
		with open(TEMP_DIR + "user_actions.csv", "a") as file:
			file.write(str(self.user_action_count))
		with open(TEMP_DIR + "episode_lengths.csv", "a") as file:
			file.write(str(self.episode_lengths[-1]))
		with open(TEMP_DIR + "episode_times.csv", "a") as file:
			file.write(str(self.episode_times[-1]))
		with open(TEMP_DIR + "mem_up_times.csv", "a") as file:
			file.write(str(self.mem_up_times[-1]))
		with open(TEMP_DIR + "train_times.csv", "a") as file:
			file.write(str(self.train_times[-1]))
		# todo action_stat
		# todo self.state_score_data


def close_file(file, save=False):
	if save:
		with open(TEMP_DIR + file, "r") as source, open(LOG_DIR + file.rstrip(".csv") + ".txt", "a") as dest:
			dest.write(ENTRY_SEPARATOR)
			line = source.readline()
			while line != "":
				dest.write(line)

	with open(TEMP_DIR + "rewards.csv", "w") as f:
		f.write("")


def update_graphs(i):
	start_time = time()
	for info in infos:
		vals = np.genfromtxt("./temp/" + info["file"], delimiter=",")
		if len(vals) < 1:
			continue
		info["plot_func"](info["axis"], vals)
		# info["axis"].set_title(info["title"], fontdict={"fontsize": 12})
	print("updating graphs took {0:.2f}sec".format(time()-start_time))


def est_errs_full_plot(axis, vals):
	s = max(vals) / 50
	bins = [0]
	for i in range(50):
		next_bin = int(i*s)
		if next_bin > bins[-1]:
			bins.append(next_bin)

	axis.clear()
	axis.hist(vals, bins, histtype="bar")


def est_errs_low_plot(axis, vals):
	bins = [i for i in range(25)]
	axis.clear()
	axis.hist(vals, bins, histtype="bar")


def rewards_plot(axis, vals):
	n_displayed = 100
	if len(vals) < n_displayed:
		x = [i for i in range(len(vals))]
		y = vals
	else:
		y = stats.average_into(vals, n_displayed)
		x = [i*(len(vals)/n_displayed) for i in range(n_displayed)]

	axis.clear()
	axis.plot(x, y)


def state_diff_plot(axis, vals):
	n_displayed = 100
	x = [i for i in range(n_displayed)]

	ys = np.array_split(vals, 10, axis=1)

	for i in range(len(ys)):
		y = stats.average_into(ys[i], n_displayed)
		axis[i].clear()
		axis[i].plot(x, y)


def q_vals_plot(axis, vals):
	n_displayed = 100
	x = [i for i in range(n_displayed)]

	ys = np.array_split(vals, 12, axis=1)

	for i in range(len(ys)):
		y = stats.average_into(ys[i], n_displayed)
		axis[i].clear()
		axis[i].plot(x, y)


if __name__ == "__main__":
	style.use("fivethirtyeight")
	figure = plt.figure()

	rows = 3
	cols = 6
	infos = [
		{"title": "Estimation Errors Full", "file": "estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 1), "plot_func": est_errs_full_plot},
		{"title": "Estimation Errors Low", "file": "estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 2), "plot_func": est_errs_low_plot},
		{"title": "Rewards", "file": "rewards.csv", "axis": figure.add_subplot(rows, cols, 3), "plot_func": rewards_plot},
		# {"title": "States Differentials", "file": "state_diffs.csv", "axis": [figure.add_subplot(rows, cols, 6 + i) for i in range(10)], "plot_func": state_diff_plot},
		{"title": "States Differentials", "file": "q_values.csv", "axis": [figure.add_subplot(rows, cols, 7 + i) for i in range(12)], "plot_func": q_vals_plot}
	]

	ani = animation.FuncAnimation(figure, update_graphs, interval=10000)
	plt.show()
