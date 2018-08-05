import numpy as np
import util.game_info as gi
from Networks.q_learning import NeuralNetwork
from PIL import Image

PROJECT_ROOT = str(__file__).replace("util/sampling.py", "")
TEMP_DIR = PROJECT_ROOT + "util/img_temp/"


def sample_net(net_id, resolution=0.01):
	net = NeuralNetwork.restore(net_id)

	width = int(gi.ARENA_WIDTH * resolution) + 1
	length = int(gi.ARENA_LENGTH * resolution) + 1
	height = int(gi.ARENA_HEIGHT * resolution) + 1
	classes = net.n_classes
	sample = np.zeros([width, length, height, classes])

	n_samples = width * length * height
	print("collecting", n_samples, "samples ...")

	# state = np.zeros(int(net.net_config["Format"]["input_shape"]))
	state = [0, 0, 0, -2, -2, -2, 0, 0, 0, 100]

	i = 0
	for x, y, z, s in sample_positions(state, 0, resolution=resolution):
		sample[x][y][z] = net.run([s])

		i += 1
		if i % 5000 == 0:
			print(str(round(100 * i / n_samples, 2)) + "%")

	return sample


def sample_positions(state, pos_start_index, resolution=0.2):
	for x in range(int(gi.ARENA_WIDTH * resolution) + 1):
		for y in range(int(gi.ARENA_LENGTH * resolution) + 1):
			for z in range(int(gi.ARENA_HEIGHT * resolution) + 1):
				state_ = state[:]
				scale = 1 / resolution
				state_[pos_start_index] = x * scale
				state_[pos_start_index + 1] = y * scale
				state_[pos_start_index + 2] = z * scale

				yield x, y, z, state_


def show_choice(sample, csv=False, img_file=True, show=False, width=True, length=True, height=False):
	w = 1
	l = 1
	h = 1
	n_actions = len(sample[0][0][0])

	if width:
		w = len(sample)
	if length:
		l = len(sample[0])
	if height:
		h = len(sample[0][0])

	selected_actions = np.zeros([h, w, l])
	for x in range(w):
		for y in range(l):
			for z in range(h):
				for a in range(n_actions):
					selected_actions[z][x][y] = np.argmax(sample[x][y][z])

	for z in range(h):
		img_array = normalize(selected_actions[z][:][:])
		img = Image.fromarray(img_array)
		if img.mode != "RGB":
			img = img.convert("RGB")

		if img_file:
			img.save(TEMP_DIR + "actions_" + str(z) + ".png")
		if csv:
			np.savetxt(TEMP_DIR + "actions.csv", img_array, delimiter=",")
		if show:
			img.show()


def show_q_values(sample, csv=False, img_file=True, show=False, width=True, length=True, height=False):
	w = 1
	l = 1
	h = 1
	n_actions = len(sample[0][0][0])

	if width:
		w = len(sample)
	if length:
		l = len(sample[0])
	if height:
		h = len(sample[0][0])

	img_data = np.zeros([n_actions, h, w, l])

	for x in range(w):
		for y in range(l):
			for z in range(h):
				for a in range(n_actions):
					img_data[a][z][x][y] = sample[x][y][z][a]

	for a in range(n_actions):
		for z in range(h):
			img_array = normalize(img_data[a][z][:][:])
			img = Image.fromarray(img_array)
			if img.mode != "RGB":
				img = img.convert("RGB")

			name = gi.ACTION_DESCRIPTIONS[gi.get_action_index(gi.get_action_states("no_flip")[a], "all")] + "_" + str(z)
			if img_file:
				img.save(TEMP_DIR + name + ".png")
			if csv:
				np.savetxt(TEMP_DIR + name + ".csv", img_array, delimiter=",")
			if show:
				img.show()


def normalize(img_data):
	v_max = -100000
	v_min = 100000
	for row in range(len(img_data)):
		v_max = max(v_max, max(img_data[row]))
		v_min = min(v_min, min(img_data[row]))

	print("min:", v_min, "max:", v_max)

	for w in range(len(img_data)):
		for h in range(len(img_data[0])):
			v = img_data[w][h]
			v = ((v - v_min) / (v_max - v_min)) * 255

			if v > 255:
				print("error", w, h)

			img_data[w][h] = v

	return img_data


if __name__ == '__main__':
	sample = sample_net("FlowBot1532874954", resolution=0.01)
	show_q_values(sample)
	show_choice(sample, csv=False, show=False, img_file=True, height=True)
