import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from util import stats, game_info as gi
from configparser import ConfigParser
from scipy.interpolate import spline

PROJECT_ROOT = str(__file__).replace("util/graphs/offline_graphs.py", "")
TEMP_DIR = PROJECT_ROOT + "util/temp/"
LOG_DIR = PROJECT_ROOT + "util/logs/"
NET_DIR = PROJECT_ROOT + "Networks/saved/"
SAVE_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Diagrams/"
VERBOSE = False

action_counts_all_tob = [1649783, 309527, 147870, 141391, 320286, 178412, 274622, 165084,
							239448, 251223, 188318, 290155, 213037, 232899, 169770, 199118,
							314126, 207407, 147208, 193214, 130571, 165308, 296463, 177198,
							177655, 237462, 240246, 213827, 169740, 202578, 149248, 117873,
							198860, 311093, 236327, 197635]


def height():
	x = [i * 20 for i in range(101)]
	y1 = [(20 * i) / 2000.0 for i in range(101)]
	y2 = [((20 * i) / 2000.0) ** 2 for i in range(101)]
	y3 = [((20 * i) / 2000.0) ** 3 for i in range(101)]
	y4 = [((20 * i) / 2000.0) ** 4 for i in range(101)]

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


def angle_to():
	x = [i - 180 for i in range(361)]
	y1 = [max(0.0, (((180 - abs(x[i])) / 180.0) - 0.5) * 2) for i in range(361)]
	y2 = []
	for i in range(361):
		if y1[i] < 0:
			y2.append(-(y1[i] ** 2))
		else:
			y2.append(y1[i] ** 2)

	y3 = []
	for i in range(361):
		y3.append(y1[i] ** 3)

	y4 = []
	for i in range(361):
		if y1[i] < 0:
			y4.append(-(y1[i] ** 4))
		else:
			y4.append(y1[i] ** 4)

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


def bool_height():
	x = [i * 20 for i in range(101)]
	y = [0 for _ in range(101)]

	y[int(len(y) / 2):] = [1 for _ in range(int(len(y) / 2) + 1)]

	plt.plot(x, y)
	plt.show()


def bool_angle():
	x = [i - 180 for i in range(361)]
	y = [0 for _ in range(361)]

	y[int(len(y) / 2) - 60: int(len(y) / 2) + 60] = [1 for _ in range(120)]

	plt.plot(x, y)
	plt.show()


def discrete_height(step_size=20):
	x = [i * 20 for i in range(101)]
	y1 = [(20 * step_size * int(i / step_size)) / 2000.0 for i in range(101)]
	y2 = [y1[i] ** 2 for i in range(101)]
	y3 = [y1[i] ** 3 for i in range(101)]
	y4 = [y1[i] ** 4 for i in range(101)]

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


def discrete_angle(step_size=20, no_neg=False):
	x = [i - 180 for i in range(361)]
	y1 = [(((180 - abs(x[i])) / 180.0) - 0.5) * 2 for i in range(361)]
	if no_neg:
		y1 = [max(0.0, y1[i]) for i in range(361)]

	n_sections = int(len(x) / step_size) + 1
	for a in range(n_sections - 1):
		y1[a * step_size: (a + 1) * step_size] = [y1[a * step_size] for _ in range(step_size)]

	x = [x[i] - step_size / 2 for i in range(361)]

	y2 = []
	for i in range(361):
		if y1[i] < 0:
			y2.append(-(y1[i] ** 2))
		else:
			y2.append(y1[i] ** 2)

	y3 = []
	for i in range(361):
		y3.append(y1[i] ** 3)

	y4 = []
	for i in range(361):
		if y1[i] < 0:
			y4.append(-(y1[i] ** 4))
		else:
			y4.append(y1[i] ** 4)

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


def reward_graph(net_name):
	print(net_name)
	y = np.loadtxt(LOG_DIR + net_name + "/reward_info.csv", delimiter=",").transpose().tolist()
	x = [i for i in range(len(y[0]))]
	plot_names = ["re_height", "re_airtime", "re_ball_dist", "re_facing_up", "re_facing_opp", "re_facing_ball"]
	figure = plt.figure()
	for i in range(6):
		axis = figure.add_subplot(2, 3, i + 1)
		axis.set_title(plot_names[i], fontdict={"fontsize": 12})
		axis.plot(x, y[i])
		print(plot_names[i] + ": " + str(round(sum(y[i]) / len(y[i]), 3)))
	if VERBOSE:
		plt.show()
	print()


def show_all(src_dir, save_dir):
	for info in infos:
		if isinstance(info["file"], list):
			vals = []
			for file in info["file"]:
				vals.append(np.loadtxt(src_dir + file, delimiter=","))
			vals = np.array(vals)
		else:
			vals = np.loadtxt(src_dir + info["file"], delimiter=",")

		if vals.shape == () or vals.shape[0] <= 1:
			print("bad values:", src_dir.split("/")[-2], info["file"])
			continue
		info["plot_func"](info["title"], vals, save_dir)


def est_errs_full_plot(title, vals, save_dir):
	s = max(vals) / 50
	bins = [0]
	for i in range(50):
		next_bin = int(i * s)
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
	vals = np.split(vals, len(vals[0]), axis=1)

	for i in range(len(vals)):
		y = stats.average_into(vals[i], n_displayed)
		x = np.linspace(0, len(y), len(y))
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
	try:
		n_qs = len(real_q_vals[0])
	except TypeError:
		print("too few q_value entries")
		return

	real_q_vals = np.array_split(real_q_vals, n_qs, axis=1)
	pred_q_vals = np.array_split(pred_q_vals, n_qs, axis=1)

	for i in range(n_qs):
		y_r = stats.average_into(real_q_vals[i], n_displayed)
		y_p = stats.average_into(pred_q_vals[i], n_displayed)
		x = [i for i in range(len(y_r))]
		plt.title(title, fontdict={"fontsize": 12})
		plt.plot(x, y_r, x, y_p)
		if VERBOSE:
			plt.show()
		plt.savefig(save_dir + "qv" + str(i) + ".png")
		plt.clf()


def simple_averaged_plot(title, vals, save_dir):
	n_points = min(len(vals), 100)
	x = [i * (len(vals) / n_points) for i in range(n_points)]
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
	try:
		plt.hist(vals[-1], [i for i in range(len(vals[0]))], histtype="bar")
	except ValueError:
		plt.hist(vals[-1], [i for i in range(len(vals[0]))], range=(0, len(vals[0])), histtype="bar")
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


def reward_comparison(bots, max_reward=1e+6, norm_len=None, norm_y=False, ref=None, n_points=100, legend=False):
	"""
	Creates a diagram with a line for each bot, displaying the reward they got throughout training
	:param bots: the id's of the bots for the diagram
	:param max_reward: cutoff point for rewards; any bot with a maximum reward greater than max_reward will not be
						included in the diagram
	:param norm_len: specifies the maximum value on the x-Axis, all lines will be streched to this length
						if None the x-Axis will go from 0 to the episode count of the longest running bot
	:param norm_y: whether the values are per iteration or per episode (for bots with task tob all episodes have the
					same length; by dividing the per-episode-value by the episode length a per-iteration-value is calculated)
	:param ref:	the reference value for the bots (how much reward random action yields in the given circumstance)
	:param n_points: how many points are displayed of each line (high variance can clutter the diagram if n_points is too high)
	:param legend: whether the legend is displayed
	:return:
	"""
	avrg_x = norm_len if norm_len is not None else 0
	averages = np.zeros([n_points])
	avrg_ep_len = 0
	for b in bots:
		description = descriptor(b, delimiter=", ")
		ep_lens = np.loadtxt(LOG_DIR + b + "/episode_lengths.csv", delimiter=",")
		ep_len = max(1, int(sum(ep_lens)/len(ep_lens)))
		avrg_ep_len += ep_len

		rewards = np.loadtxt(LOG_DIR + b + "/reward_info.csv", delimiter=",")
		rewards = np.sum(rewards, axis=1)
		rewards = reduce_rewards(rewards, ep_len, norm=norm_y)

		if max(rewards) > max_reward:
			continue

		print(b)
		print(description)
		m_index = np.argmax(rewards)
		print("Max: {0:.2f} at {1:d} ({2:d}|{3:.2f}%)".format(max(rewards), m_index, len(rewards), 100 * m_index / len(rewards)))
		print("Last:", rewards[-1])
		print(stats.DistributionInfo(rewards))
		print()

		x_max = len(rewards) if norm_len is None else norm_len
		if x_max > avrg_x:
			avrg_x = x_max
		x_old = np.linspace(0, x_max, len(rewards))
		x_new = np.linspace(0, x_max, n_points)
		smoothed = spline(x_old, rewards, x_new)
		plt.plot(x_new, smoothed, label=description)

		for i in range(n_points):
			averages[i] += smoothed[i]

	avrg_ep_len /= len(bots)
	avrg_x = np.linspace(0, avrg_x, n_points)
	averages = [val / len(bots) for val in averages]
	plt.plot(avrg_x, averages, label="average", color="k", linestyle="--")

	if ref is not None:
		if norm_y:
			ref_y = [ref for _ in range(n_points)]
		else:
			ref_y = [ref*avrg_ep_len for _ in range(n_points)]

		plt.plot(avrg_x, ref_y, label="reference", color="k")
	if legend:
		plt.legend(loc="upper center", fontsize=12, framealpha=1)
	plt.show()
	plt.clf()


def action_summation(bots, n_actions):
	action_counts = np.zeros([n_actions])
	for i, bot_name in enumerate(bots):
		data = np.loadtxt(LOG_DIR + bot_name + "/net_output.csv", delimiter=",")
		for i in range(len(data)):
			action_counts[np.argmax(data[i])] += 1
		print(str(i/len(bots)) + "%")

	print(action_counts)

	x = np.linspace(0, len(action_counts), len(action_counts))
	plt.bar(x, action_counts)
	plt.show()
	plt.clf()


def avrg_episode_length(bots):
	data = []
	for bot_name in bots:
		ep_lens = np.loadtxt(LOG_DIR + bot_name + "/episode_times.csv")
		data.append(ep_lens)

	max_len = max([len(row) for row in data])
	'''
	y = np.zeros([max_len])
	n = np.zeros([max_len])

	for r in range(len(data)):
		for c in range(len(data[r])):
			y[c] += data[r][c]
			n[c] += 1

	for i in range(len(y)):
		y[i] /= n[i]

	print(y)
	print(n)
	'''

	for y in data:
		x = [i for i in range(len(y))]
		plt.plot(x, y)
	plt.show()
	plt.clf()


def rare_actions(bots, n_actions):
	for i, bot_name in enumerate(bots):
		rares = np.zeros([n_actions])
		data = np.loadtxt(LOG_DIR + bot_name + "/net_output.csv", delimiter=",")
		for x in range(len(data)):
			for y in range(len(data[x])):
				if data[x][y] == 0:
					rares[y] += 1

		print(bot_name, rares)
		print()


def reduce_rewards(rewards, ep_len, norm=False):
	res = []
	n_sections = int(len(rewards)/ep_len)
	for n in range(n_sections):
		start = n*ep_len
		end = start + ep_len
		res.append(sum(rewards[start:end]))
		if norm:
			res[-1] /= ep_len
	return res


def get_bots(net_type=None, bot_type=None, task=None, sarsa=None, neg_reward=None, include_reference=False):
	reader = ConfigParser()
	reader.read(LOG_DIR + "run_index.cfg")
	bots = []
	for bot_name in reader.keys():
		if not include_reference and re.search("ref", bot_name) is not None:
			continue

		try:
			d = descriptor(bot_name).split(":")
		except BadBotError as e:
			print(e)
			continue

		match_nt = net_type is None or d[0] == net_type
		match_bt = bot_type is None or d[1] == bot_type
		match_t = task is None or d[2] == task
		match_s = sarsa is None or d[3] == sarsa
		match_nr = neg_reward is None or d[4] == neg_reward

		if match_nt and match_bt and match_t and match_s and match_nr:
			print(bot_name, d)
			bots.append(bot_name)

	if len(bots) == 0:
		print("no bots found")
	return bots


def fix_neg_reward():
	reader = ConfigParser()
	reader.read(LOG_DIR + "run_index.cfg")

	for bot_name in reader.keys():
		try:
			_ = reader[bot_name]["neg_reward"]
		except KeyError:
			try:
				rewards = np.loadtxt(LOG_DIR + bot_name + "/reward_info.csv", delimiter=",")
				min_reward = np.min(rewards)
				if min_reward < 0:
					reader[bot_name]["neg_reward"] = "True"
				else:
					reader[bot_name]["neg_reward"] = "False"
			except OSError:
				print("could not decide neg reward for", bot_name)

	with open(LOG_DIR + "run_index_corrected.cfg", "w") as file:
		reader.write(file)


def descriptor(bot_name, delimiter=":"):
	run_index = ConfigParser()
	run_index.read(LOG_DIR + "run_index.cfg")
	try:
		bot_info = run_index[bot_name]
	except KeyError:
		raise BadBotError(bot_name, "not found")

	try:
		end_conditions = bot_info["end_conditions"].split(", ")
	except KeyError:
		raise BadBotError(bot_name, "No end condition info")
	if end_conditions[0] != "None":
		task = "tob"
	else:
		task = "fly"

	try:
		sarsa = "s" if bot_info["sarsa"] == "True" else "x"
	except KeyError:
		raise BadBotError(bot_name, "No sarsa info")

	try:
		neg_reward = "n" if bot_info["neg_reward"] == "True" else "x"
	except KeyError:
		try:
			rewards = np.loadtxt(LOG_DIR + bot_name + "/reward_info.csv", delimiter=",")
		except OSError:
			raise BadBotError(bot_name, "no reward info")

		min_reward = np.min(rewards)
		neg_reward = "n" if min_reward < 0 else "x"

	bot_type = bot_info["bot_type"]

	return net_descriptor(bot_name) + delimiter + bot_type + delimiter + task + delimiter + sarsa + delimiter + neg_reward


def net_descriptor(bot_name):
	net_cfg = ConfigParser()
	net_cfg.read(NET_DIR + bot_name + "/net.cfg")

	net_format = ""
	if int(net_cfg["Layer0"]["size"]) == 256:
		net_format += "c"
	else:
		net_format += "f"

	size = int(net_cfg["Format"]["n_layers"])
	has_do = not size == 1 and net_cfg["Layer1"]["type"] == "do"
	if has_do:
		size /= 2

	net_format += str(int(size))
	if has_do:
		net_format += "d"

	return net_format


def create_graphs(bots=None):
	if bots is None:
		bots = get_bots()

	for i, bot in enumerate(bots):
		print("{0:d} out of {1:d} completed".format(i, len(bots)))
		print("current:", bot)
		desc = descriptor(bot, delimiter=",")
		_, _, task, _, _ = desc.split(",")

		src_dir = LOG_DIR + bot + "/"
		save_dir = SAVE_DIR + "task " + task + "/" + desc + " - " + bot + "/"
		if os.path.isdir(save_dir):
			print("graphs already exist\n")
			continue
		else:
			os.makedirs(save_dir)

		show_all(src_dir, save_dir)


def create_graph(bot_name):
	desc = descriptor(bot_name, delimiter=",")
	_, _, task, _, _ = desc.split(",")

	src_dir = LOG_DIR + bot_name + "/"
	save_dir = SAVE_DIR + "task " + task + "/" + desc + " - " + bot_name + "/"

	if os.path.isdir(save_dir):
		print("graphs already exist")
		return
	else:
		os.makedirs(save_dir)

	show_all(src_dir, save_dir)


class BadBotError(Exception):
	def __init__(self, bot_name, reason):
		self.reason = reason
		self.bot = bot_name

	def __str__(self):
		return self.bot + " was not valid because: " + self.reason


if __name__ == '__main__':
	style.use("fivethirtyeight")
	figure = plt.figure()

	rows = 3
	cols = 4
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

	bots_taxx = ["FlowBot_tob1535467552", 'FlowBot1536231138', 'FlowBot1536232921', 'FlowBot1536235613', 'FlowBot1536238030', 'FlowBot1536240328', 'FlowBot1536241818', 'FlowBot1536268211']

	bot_type = "all"
	bots = get_bots(net_type=None, bot_type=bot_type, task="tob", sarsa="s", neg_reward="n")

	bots.append("FlowBot1536677070_a")

	# create_graphs(["FlowBot1536677070_a"])
	# create_graph("FlowBot1536268211")

	ref = 0.0 ** 2
	reward_comparison(bots, norm_len=100, norm_y=True, n_points=25, ref=ref, legend=True)

	# action_summation(bots, n_actions=len(gi.get_action_states(bot_type)))
	# rare_actions(bots, n_actions=len(gi.get_action_states(bot_type)))
	# avrg_episode_length(bots)

	# fix_neg_reward()
