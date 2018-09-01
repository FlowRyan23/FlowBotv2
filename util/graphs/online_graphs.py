import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from time import time
from util import stats

PROJECT_ROOT = str(__file__).replace("util/graphs/online_graphs.py", "")
TEMP_DIR = PROJECT_ROOT + "util/temp/"
LOG_DIR = PROJECT_ROOT + "util/logs/"


def update_graphs(i):
	start_time = time()
	for info in infos:
		if isinstance(info["file"], list):
			vals = []
			for file in info["file"]:
				vals.append(np.genfromtxt(src_dir + file, delimiter=","))
		else:
			vals = np.genfromtxt(src_dir + info["file"], delimiter=",")
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

	real_q_vals = vals[0]
	pred_q_vals = vals[1]
	n_qs = len(real_q_vals[0])

	real_q_vals = np.array_split(real_q_vals, n_qs, axis=1)
	pred_q_vals = np.array_split(pred_q_vals, n_qs, axis=1)

	for i in range(n_qs):
		y_r = stats.average_into(real_q_vals[i], n_displayed)
		y_p = stats.average_into(pred_q_vals[i], n_displayed)
		x = [i for i in range(len(y_r))]
		axis[i].clear()
		axis[i].plot(x, y_r, y_p)


def simple_averaged_plot(axis, vals):
	n_points = min(len(vals), 100)
	x = [i * (len(vals)/n_points) for i in range(n_points)]
	y = stats.average_into(vals, n_points)

	axis.clear()
	axis.plot(x, y)


def net_output_plot(axis, vals):
	net_plot_helper(axis[0], vals)

	start = min(len(vals) - 1, 100)
	net_plot_helper(axis[1], vals[-start:-1])

	start = min(len(vals) - 1, 10)
	net_plot_helper(axis[2], vals[-start:-1])

	axis[3].clear()
	axis[3].hist(vals[-1], [i for i in range(len(vals[0]))], histtype="bar")


def net_plot_helper(axis, vals):
	y = []
	for a in range(len(vals[0])):
		y_cur = 0
		for iter in range(len(vals)):
			y_cur += vals[iter][a]
		y.append(y_cur / len(vals))

	x = [i for i in range(len(y))]
	axis.clear()
	axis.bar(x, y)


if __name__ == '__main__':
	style.use("fivethirtyeight")
	figure = plt.figure()
	src_dir = TEMP_DIR
	update_interval = 20000

	rows = 3
	cols = 4
	n_actions = 18
	infos = [
		{"title": "Estimation Errors Full", "file": "estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 1), "plot_func": est_errs_full_plot},
		{"title": "Estimation Errors Low", "file": "estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 2), "plot_func": est_errs_low_plot},
		{"title": "Averaged Estimation Errors", "file": "avrg_estimation_errors.csv", "axis": figure.add_subplot(rows, cols, 3), "plot_func": simple_averaged_plot},
		{"title": "Rewards", "file": "rewards.csv", "axis": figure.add_subplot(rows, cols, 4), "plot_func": simple_averaged_plot},
		{"title": "Iterations per Episode", "file": "episode_lengths.csv", "axis": figure.add_subplot(rows, cols, 5), "plot_func": simple_averaged_plot},
		{"title": "Episode length", "file": "episode_times.csv", "axis": figure.add_subplot(rows, cols, 6), "plot_func": simple_averaged_plot},
		{"title": "Q_Update length", "file": "mem_up_times.csv", "axis": figure.add_subplot(rows, cols, 7), "plot_func": simple_averaged_plot},
		{"title": "Training lenght", "file": "train_times.csv", "axis": figure.add_subplot(rows, cols, 8), "plot_func": simple_averaged_plot},
		{"title": "Net Output", "file": "net_output.csv", "axis": [figure.add_subplot(rows, cols, 9 + i) for i in range(4)], "plot_func": net_output_plot},
		# {"title": "States Differentials", "file": "state_diffs.csv", "axis": [figure.add_subplot(rows, cols, 13 + i) for i in range(10)], "plot_func": state_diff_plot},
		# {"title": "Q-Values", "file": ["q_values.csv", "pred_q_values.csv"], "axis": [figure.add_subplot(rows, cols, 13 + i) for i in range(n_actions)], "plot_func": q_vals_plot}
	]

	ani = animation.FuncAnimation(figure, update_graphs, interval=update_interval)
	plt.show()