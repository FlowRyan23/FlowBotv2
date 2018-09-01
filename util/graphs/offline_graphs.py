import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from util import stats
from configparser import ConfigParser
from scipy.interpolate import spline

PROJECT_ROOT = str(__file__).replace("util/graphs/offline_graphs.py", "")
TEMP_DIR = PROJECT_ROOT + "util/temp/"
LOG_DIR = PROJECT_ROOT + "util/logs/"
SAVE_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Diagrams/"
VERBOSE = False


def reward_graph(net_name):
	print(net_name)
	y = np.loadtxt(LOG_DIR + net_name + "/reward_info.csv", delimiter=",").transpose().tolist()
	x = [i for i in range(len(y[0]))]
	plot_names = ["re_height", "re_airtime", "re_ball_dist", "re_facing_up", "re_facing_opp", "re_facing_ball"]
	figure = plt.figure()
	for i in range(6):
		axis = figure.add_subplot(2, 3, i+1)
		axis.set_title(plot_names[i], fontdict={"fontsize": 12})
		axis.plot(x, y[i])
		print(plot_names[i] + ": " + str(round(sum(y[i])/len(y[i]), 3)))
	if VERBOSE:
		plt.show()
	print()


def show_all(src_dir, save_dir):
	for info in infos:
		if isinstance(info["file"], list):
			vals = []
			for file in info["file"]:
				vals.append(np.genfromtxt(src_dir + file, delimiter=","))
		else:
			vals = np.genfromtxt(src_dir + info["file"], delimiter=",")
		if len(vals) < 1:
			continue
		info["plot_func"](info["title"], vals, save_dir)


def est_errs_full_plot(title, vals, save_dir):
	s = max(vals) / 50
	bins = [0]
	for i in range(50):
		next_bin = int(i*s)
		if next_bin > bins[-1]:
			bins.append(next_bin)
			
	plt.title(title, fontdict={"fontsize": 12})
	plt.hist(vals, bins, histtype="bar")
	if VERBOSE:
		plt.show()
	plt.savefig(save_dir + "est errs full")
	plt.clf()


def est_errs_low_plot(title, vals, save_dir):
	bins = [i for i in range(25)]
	plt.title(title, fontdict={"fontsize": 12})
	plt.hist(vals, bins, histtype="bar")
	if VERBOSE:
		plt.show()
	plt.savefig(save_dir + "est errs low.png")
	plt.clf()


def state_diff_plot(title, vals, save_dir):
	n_displayed = 100
	x = [i for i in range(n_displayed)]

	vals = np.split(vals, len(vals[0]), axis=1)

	for i in range(len(vals)):
		y = stats.average_into(vals[i], n_displayed)
		plt.title(title, fontdict={"fontsize": 12})
		plt.plot(x, y)
		if VERBOSE:
			plt.show()
		plt.savefig(save_dir + "sd" + str(i) + ".png")
		plt.clf()


def q_vals_plot(title, vals, save_dir):
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
		plt.title(title, fontdict={"fontsize": 12})
		plt.plot(x, y_r, y_p)
		if VERBOSE:
			plt.show()
		plt.savefig(save_dir + "qv" + str(i) + ".png")
		plt.clf()


def simple_averaged_plot(title, vals, save_dir):
	n_points = min(len(vals), 100)
	x = [i * (len(vals)/n_points) for i in range(n_points)]
	y = stats.average_into(vals, n_points)

	plt.title(title, fontdict={"fontsize": 12})
	plt.plot(x, y)
	if VERBOSE:
		plt.show()
	plt.savefig(save_dir + title + ".png")
	plt.clf()


def net_output_plot(title, vals, save_dir):
	net_plot_helper(title, vals)
	plt.savefig(save_dir + "actions full.png")
	plt.clf()

	start = min(len(vals) - 1, 100)
	net_plot_helper(title, vals[-start:-1])
	plt.savefig(save_dir + "actions 100.png")
	plt.clf()

	start = min(len(vals) - 1, 10)
	net_plot_helper(title, vals[-start:-1])
	plt.savefig(save_dir + "actions 10.png")
	plt.clf()

	plt.title(title, fontdict={"fontsize": 12})
	plt.hist(vals[-1], [i for i in range(len(vals[0]))], histtype="bar")
	if VERBOSE:
		plt.show()


def net_plot_helper(title, vals):
	y = []
	for a in range(len(vals[0])):
		y_cur = 0
		for iter in range(len(vals)):
			y_cur += vals[iter][a]
		y.append(y_cur / len(vals))

	x = [i for i in range(len(y))]
	plt.title(title, fontdict={"fontsize": 12})
	plt.bar(x, y)
	if VERBOSE:
		plt.show()


def reward_comparison(bots, graph=True, max_reward=1e+8, norm_len=100):
	for a in bots:
		rewards = np.loadtxt(LOG_DIR + a + "/rewards.csv", delimiter=",")
		if max(rewards) > max_reward:
			continue
		print(a)
		m_index = np.argmax(rewards)
		print("Max: {0:.2f} at {1:d} ({2:d}|{3:.2f}%)".format(max(rewards), m_index, len(rewards), 100 * m_index/len(rewards)))
		print("Last:", rewards[-1])
		print(stats.DistributionInfo(rewards))
		print()

		if graph:  # and 100 * m_index/len(rewards) > 70:
			n_points = 10
			x_max = len(rewards) if norm_len is None else norm_len
			x_old = np.linspace(0, x_max, len(rewards))
			x_new = np.linspace(0, x_max, n_points)
			smoothed = spline(x_old, rewards, x_new)
			plt.plot(x_new, smoothed)
	if graph:
		plt.show()
		plt.clf()


def get_bots(s):
	reader = ConfigParser()
	reader.read(LOG_DIR + "run_index.cfg")
	bots = []
	for key in reader.keys():
		if re.search(s, key):
			bots.append(key)
	return bots


if __name__ == '__main__':
	style.use("fivethirtyeight")
	figure = plt.figure()
	bot_name = "FlowBot_fly1535546263"
	src_dir = LOG_DIR + bot_name + "/"
	save_dir = SAVE_DIR + bot_name + "/"
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	rows = 3
	cols = 4
	n_actions = 18
	infos = [
		{"title": "Estimation Errors Full", "file": "estimation_errors.csv", "plot_func": est_errs_full_plot},
		{"title": "Estimation Errors Low", "file": "estimation_errors.csv", "plot_func": est_errs_low_plot},
		{"title": "Averaged Estimation Errors", "file": "avrg_estimation_errors.csv", "plot_func": simple_averaged_plot},
		{"title": "Rewards", "file": "rewards.csv", "plot_func": simple_averaged_plot},
		{"title": "Iterations per Episode", "file": "episode_lengths.csv", "plot_func": simple_averaged_plot},
		{"title": "Episode length", "file": "episode_times.csv", "plot_func": simple_averaged_plot},
		{"title": "Q_Update length", "file": "mem_up_times.csv", "plot_func": simple_averaged_plot},
		{"title": "Training lenght", "file": "train_times.csv", "plot_func": simple_averaged_plot},
		{"title": "Net Output", "file": "net_output.csv", "plot_func": net_output_plot},
		{"title": "States Differentials", "file": "state_diffs.csv", "plot_func": state_diff_plot},
		{"title": "Q-Values", "file": ["q_values.csv", "pred_q_values.csv"], "plot_func": q_vals_plot}
	]

	# show_all(src_dir, save_dir)
	reward_comparison(get_bots("tob"))
