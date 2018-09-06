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
NET_DIR = PROJECT_ROOT + "Networks/saved/"
SAVE_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Diagrams/"
VERBOSE = False

ref_e1 = [
	[0.58138, 0.00000, 0.66332, 0.07746, 0.47645, 0.38730],
	[0.17321, 0.04472, 0.81363, 0.33466, 0.45717, 0.47958],
	[0.27386, 0.04472, 0.68118, 0.23238, 0.49900, 0.41231],
	[0.23664, 0.04472, 0.76158, 0.47117, 0.45497, 0.39497],
	[0.56480, 0.03162, 0.77846, 0.15492, 0.47329, 0.45277],
	[0.27019, 0.04472, 0.69282, 0.19494, 0.44272, 0.35071],
	[0.20000, 0.04472, 0.82158, 0.31464, 0.45277, 0.44609],
	[0.08367, 0.00000, 0.94181, 0.00000, 0.13784, 0.05477]
]
ref_e1_ulb = [
	[0.62530, 0.03162, 0.76223, 0.13416, 0.42426, 0.43932],
	[0.41110, 0.04472, 0.81486, 0.27019, 0.42661, 0.41231],
	[0.55857, 0.04472, 0.76616, 0.26268, 0.45166, 0.39243],
	[0.37014, 0.04472, 0.80250, 0.37683, 0.43012, 0.41473],
	[0.54955, 0.03162, 0.70569, 0.15166, 0.46904, 0.44159],
	[0.53666, 0.04472, 0.70427, 0.20736, 0.48166, 0.43128],
	[0.42895, 0.04472, 0.78677, 0.27928, 0.44721, 0.40743],
	[0.05477, 0.00000, 0.51769, 0.04472, 0.55588, 0.20000]
]
ref_e2 = [
	[0.33800, 0.00000, 0.44000, 0.00600, 0.22700, 0.15000],
	[0.03000, 0.00200, 0.66200, 0.11200, 0.20900, 0.23000],
	[0.07500, 0.00200, 0.46400, 0.05400, 0.24900, 0.17000],
	[0.05600, 0.00200, 0.58000, 0.22200, 0.20700, 0.15600],
	[0.31900, 0.00100, 0.60600, 0.02400, 0.22400, 0.20500],
	[0.07300, 0.00200, 0.48000, 0.03800, 0.19600, 0.12300],
	[0.04000, 0.00200, 0.67500, 0.09900, 0.20500, 0.19900],
	[0.00700, 0.00000, 0.88700, 0.00000, 0.01900, 0.00300]
]
ref_e2_ulb = [
	[0.39100, 0.00100, 0.58100, 0.01800, 0.18000, 0.19300],
	[0.16900, 0.00200, 0.66400, 0.07300, 0.18200, 0.17000],
	[0.31200, 0.00200, 0.58700, 0.06900, 0.20400, 0.15400],
	[0.13700, 0.00200, 0.64400, 0.14200, 0.18500, 0.17200],
	[0.30200, 0.00100, 0.49800, 0.02300, 0.22000, 0.19500],
	[0.28800, 0.00200, 0.49600, 0.04300, 0.23200, 0.18600],
	[0.18400, 0.00200, 0.61900, 0.07800, 0.20000, 0.16600],
	[0.00300, 0.00000, 0.26800, 0.00200, 0.30900, 0.04000]
]
ref_e3 = [
	[0.19651, 0.00000, 0.29186, 0.00046, 0.10815, 0.05809],
	[0.00520, 0.00009, 0.53863, 0.03748, 0.09555, 0.11030],
	[0.02054, 0.00009, 0.31607, 0.01255, 0.12425, 0.07009],
	[0.01325, 0.00009, 0.44171, 0.10460, 0.09418, 0.06162],
	[0.18017, 0.00003, 0.47175, 0.00372, 0.10602, 0.09282],
	[0.01972, 0.00009, 0.33255, 0.00741, 0.08677, 0.04314],
	[0.00800, 0.00009, 0.55457, 0.03115, 0.09282, 0.08877],
	[0.00059, 0.00000, 0.83538, 0.00000, 0.00262, 0.00016]
]
ref_e3_ulb = [
	[0.24449, 0.00003, 0.44286, 0.00241, 0.07637, 0.08479],
	[0.06948, 0.00009, 0.54107, 0.01972, 0.07764, 0.07009],
	[0.17427, 0.00009, 0.44974, 0.01812, 0.09214, 0.06043],
	[0.05071, 0.00009, 0.51681, 0.05351, 0.07957, 0.07133],
	[0.16596, 0.00003, 0.35143, 0.00349, 0.10319, 0.08611],
	[0.15456, 0.00009, 0.34932, 0.00892, 0.11175, 0.08022],
	[0.07893, 0.00009, 0.48701, 0.02178, 0.08944, 0.06763],
	[0.00016, 0.00000, 0.13874, 0.00009, 0.17177, 0.00800]
]
ref_e4 = [
	[0.11424, 0.00000, 0.19360, 0.00004, 0.05153, 0.02250],
	[0.00090, 0.00000, 0.43824, 0.01254, 0.04368, 0.05290],
	[0.00563, 0.00000, 0.21530, 0.00292, 0.06200, 0.02890],
	[0.00314, 0.00000, 0.33640, 0.04928, 0.04285, 0.02434],
	[0.10176, 0.00000, 0.36724, 0.00058, 0.05018, 0.04203],
	[0.00533, 0.00000, 0.23040, 0.00144, 0.03842, 0.01513],
	[0.00160, 0.00000, 0.45563, 0.00980, 0.04203, 0.03960],
	[0.00005, 0.00000, 0.78677, 0.00000, 0.00036, 0.00000]
]
ref_e4_ulb = [
	[0.15288, 0.00000, 0.33756, 0.00032, 0.03240, 0.03725],
	[0.02856, 0.00000, 0.44090, 0.00533, 0.03312, 0.02890],
	[0.09734, 0.00000, 0.34457, 0.00476, 0.04162, 0.02372],
	[0.01877, 0.00000, 0.41474, 0.02016, 0.03423, 0.02958],
	[0.09120, 0.00000, 0.24800, 0.00053, 0.04840, 0.03803],
	[0.08294, 0.00000, 0.24602, 0.00185, 0.05382, 0.03460],
	[0.03386, 0.00000, 0.38316, 0.00608, 0.04000, 0.02756],
	[0.00001, 0.00000, 0.07182, 0.00000, 0.09548, 0.00160]
]


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


def reward_comparison(bots, max_reward=1e+6, norm_len=None, ref=None, n_points=100):
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
		task = description.split(", ")[2]
		rewards = reduce_rewards(rewards, ep_len, norm=task == "tob")

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
		plt.plot(x_new, smoothed, label=b)

		for i in range(n_points):
			averages[i] += smoothed[i]

	avrg_ep_len /= len(bots)
	avrg_x = np.linspace(0, avrg_x, n_points)
	averages = [val / len(bots) for val in averages]
	plt.plot(avrg_x, averages, label="average", color="k", linestyle="--")

	if ref is not None:
		ref_y = [ref*avrg_ep_len for _ in range(n_points)]
		plt.plot(avrg_x, ref_y, label="reference", color="k")
	# plt.legend(loc="upper left", fontsize=12)
	plt.show()
	plt.clf()


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
			bots.append(bot_name)

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


def create_graphs():
	bots = get_bots()

	for i, bot in enumerate(bots):
		print("{0:d} out of {1:d} completed".format(i, len(bots)))
		print("current:", bot)
		desc = descriptor(bot, delimiter=",")
		_, _, task, _, _ = desc.split(":")

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
	_, bot_type, task, sarsa, neg_reward = desc.split(",")
	sarsa = sarsa == "s"
	neg_reward = neg_reward == "n"

	if sarsa:
		if neg_reward:
			sub_folder = "sarsa neg/"
		else:
			sub_folder = "sarsa/"
	else:
		if neg_reward:
			sub_folder = "neg/"
		else:
			sub_folder = "base/"

	src_dir = LOG_DIR + bot_name + "/"
	save_dir = SAVE_DIR + "task " + task + "/" + bot_type + "/" + sub_folder + desc + " - " + bot_name + "/"

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

	bots = get_bots(net_type=None, bot_type=None, task="tob", sarsa=None, neg_reward=None)

	# create_graphs()
	# create_graph("FlowBot_tob1535640892")

	ref = None
	reward_comparison(bots, norm_len=100, n_points=20, ref=ref)

	# fix_neg_reward()
