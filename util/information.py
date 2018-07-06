import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import util.stats as stats
from matplotlib import style
from time import time
from util.game_info import as_to_str

TEMP_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/temp/"
LOG_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/logs/"
ENTRY_SEPARATOR = "\n::Next Entry::\n"


class RunInfo:
	def __init__(self):
		self.net = None
		self.run_start_time = time()

		self.iteration_count = 0
		self.last_iter_time = time()
		self.action_stat = {}
		self.user_action_count = 0
		self.net_output = []

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

		self.last_ep_iter = self.iteration_count
		self.last_ep_time = time()

	def iteration(self, net_output, action, user=False, verbose=False):
		self.net_output.append(net_output)

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
			file.write(str(self.user_action_count) + "\n")
		with open(TEMP_DIR + "episode_lengths.csv", "a") as file:
			file.write(str(self.episode_lengths[-1]) + "\n")
		with open(TEMP_DIR + "episode_times.csv", "a") as file:
			file.write(str(self.episode_times[-1]) + "\n")
		with open(TEMP_DIR + "mem_up_times.csv", "a") as file:
			file.write(str(self.mem_up_times[-1]) + "\n")
		with open(TEMP_DIR + "train_times.csv", "a") as file:
			file.write(str(self.train_times[-1]) + "\n")
		with open(TEMP_DIR + "net_output.csv", "a") as file:
			for entry in self.net_output:
				file.write(str(entry) + "\n")
			self.net_output = []
		# todo action_stat
		# todo self.state_score_data


def update_graphs(i):
	start_time = time()
	for info in infos:
		vals = np.genfromtxt("./temp/" + info["file"], delimiter=",")
		if len(vals) < 1:
			continue
		info["plot_func"](info["axis"], vals)
		try:
			info["axis"].set_title(info["title"], fontdict={"fontsize": 12})
		except AttributeError:
			for a in info["axis"]:
				a.set_title(info["title"], fontdict={"fontsize": 12})
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

	ys = np.array_split(vals, 12, axis=1)

	for i in range(len(ys)):
		y = stats.average_into(ys[i], n_displayed)
		x = [i for i in range(len(y))]
		axis[i].clear()
		axis[i].plot(x, y)


# todo x-Axis gets wrong names
def simple_averaged_plot(axis, vals):
	n_points = min(len(vals), 100)
	x = [i for i in range(n_points)]
	y = stats.average_into(vals, n_points)
	axis.clear()
	axis.plot(x, y)


def simple_plot(axis, vals):
	x = [i for i in range(len(vals))]
	axis.clear()
	axis.plot(x, vals)


if __name__ == "__main__":
	diagrams = True
	read_file = False
	plot_test = False

	if diagrams:
		style.use("fivethirtyeight")
		figure = plt.figure()

		rows = 4
		cols = 6
		infos = [
			{"title": "Estimation Errors Full", "file": "estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 1), "plot_func": est_errs_full_plot},
			{"title": "Estimation Errors Low", "file": "estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 2), "plot_func": est_errs_low_plot},
			{"title": "Averaged Estimation Errors", "file": "avrg_estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 3), "plot_func": simple_averaged_plot},
			{"title": "Rewards", "file": "rewards.csv", "axis": figure.add_subplot(rows, cols, 4), "plot_func": rewards_plot},
			{"title": "Iterations per Episode", "file": "episode_lengths.csv", "axis": figure.add_subplot(rows, cols, 5), "plot_func": simple_averaged_plot},
			{"title": "Episode length", "file": "episode_times.csv", "axis": figure.add_subplot(rows, cols, 6), "plot_func": simple_averaged_plot},
			{"title": "Q_Update length", "file": "mem_up_times.csv", "axis": figure.add_subplot(rows, cols, 7), "plot_func": simple_averaged_plot},
			{"title": "Training lenght", "file": "train_times.csv", "axis": figure.add_subplot(rows, cols, 8), "plot_func": simple_averaged_plot},
			# {"title": "States Differentials", "file": "state_diffs.csv", "axis": [figure.add_subplot(rows, cols, 13 + i) for i in range(10)], "plot_func": state_diff_plot},
			{"title": "Q-Values", "file": "q_values.csv", "axis": [figure.add_subplot(rows, cols, 13 + i) for i in range(12)], "plot_func": q_vals_plot}
		]

		ani = animation.FuncAnimation(figure, update_graphs, interval=2000)
		plt.show()

	if read_file:
		q_vals = np.genfromtxt("./logs/FlowBot5B3A33F4/q_values.csv", dtype=np.float32, delimiter=",")
		stat = [0 for _ in range(len(q_vals[0]))]
		for i in range(len(q_vals)):
			stat[np.argmax(q_vals[i])] += 1
		print(stat)

		# for i in range(len(q_vals[0])):
		#	print(int(sum([q_vals[a][i] for a in range(len(q_vals))])))

	if plot_test:
		from random import randrange
		x = [i for i in range(50)]
		y = [randrange(0, 50) for a in range(250)]

		averaged_y = stats.average_into(y, 50)
		plt.plot(x, averaged_y)
		plt.show()

