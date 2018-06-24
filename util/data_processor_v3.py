import numpy as np
import re
import util.vector_math as vmath
import util.game_info as gi
from Networks.nn_provider import NeuralNetwork
from util.stats import DistributionInfo
from rlbot.agents.base_agent import SimpleControllerState


def action_counts(in_file, out_file, with_user=False):
	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Agent Action Count", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")

	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Teacher Action Count", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")

	if with_user:
		with open(in_file, "r") as file, open(out_file, "a") as out_file:
			line = file.readline()
			while line != "":
				if re.search("User Action Count", line) is not None:
					out_file.write(line)
				line = file.readline()
			out_file.write("\n")


def action_stats(in_file, out_file, with_user=False):
	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Agent Output Statistic", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")

	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Teacher Output Statistic", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")

	if with_user:
		with open(in_file, "r") as file, open(out_file, "a") as out_file:
			line = file.readline()
			while line != "":
				if re.search("User Output Statistic", line) is not None:
					out_file.write(line)
				line = file.readline()
			out_file.write("\n")


def fitness(in_file, out_file):
	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Average Fitness", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")


def training_data_stat(in_file, out_file):
	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Output Statistic:", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")


def population(in_file, out_file):
	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Killed", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")

	with open(in_file, "r") as file, open(out_file, "a") as out_file:
		line = file.readline()
		while line != "":
			if re.search("Survivors:", line) is not None:
				out_file.write(line)
			line = file.readline()
		out_file.write("\n")


def convert_v2_to_v3(v2_features):
	v3_features = []
	count = 0

	for feature in v2_features:
		game_data = feature[0]
		controller_data = feature[1]

		# game data conversion
		ball_info = game_data[0:9]
		self_info = game_data[15:24]
		self_info.append(game_data[27])		# boost
		opp_info = game_data[28:37]
		opp_info.append(game_data[40])		# boost

		game_data_v3 = ball_info + self_info + opp_info

		# controller data conversion
		left = right = 0
		if controller_data[0] < 0:
			left = 1
		elif controller_data[0] != 128:
			right = 1

		up = down = 0
		if controller_data[0] < 0:
			down = 1
		elif controller_data[0] != 128:
			up = 1

		acc = 0
		if controller_data[4] > 10:
			acc = 1

		decc = 0
		if controller_data[5] > 10:
			decc = 1

		boost = 0
		if controller_data[8] > 0:
			boost = 1

		jump = 0
		if controller_data[9] > 0:
			jump = 1

		slide = 0
		if controller_data[12] > 0:
			slide = 1

		controller_data_v3 = [left, right, up, down, acc, decc, jump, boost, slide]

		v3_features.append([game_data_v3, controller_data_v3])
		count += 1

	return v3_features, count


# requires the file at 'path' to in post processing format
def read_all_v2(path):
	features = []
	with open(path, 'r') as data_file:
		# read initial feature
		feature = data_file.readline()
		feature_count = 0
		while feature != '':
			feature_count += 1
			# split the feature string in x and y component
			x_y_tuple = feature.replace('\n', '').split('::::')
			x = convert_to_float_list(x_y_tuple[0])
			y = convert_to_float_list(x_y_tuple[1])
			if len(x) == 41 and len(y) == 14:
				features.append([x, y])
			else:
				print('read invalid feature at: ', feature_count)

			feature = data_file.readline()
	return features, feature_count


def read_all_v3(path, n_in, n_out):
	features = []
	with open(path, 'r') as data_file:
		# read initial feature
		feature = data_file.readline()
		feature_count = 0
		while feature != '':
			feature_count += 1
			# split the feature string in x and y component
			x_y_tuple = feature.replace('\n', '').split('::::')
			x = convert_to_float_list(x_y_tuple[0])
			y = convert_to_float_list(x_y_tuple[1])

			if len(x) == n_in and len(y) == n_out:
				features.append([x, y])
			else:
				print('read invalid feature at: ', feature_count)

			feature = data_file.readline()

	if feature_count <= 0:
		raise ValueError("no features were found")

	return features, feature_count


def write_all(features, dest_path):
	with open(dest_path, 'a') as file:
		for feature in features:
			as_string = convert_to_feature_string(feature)
			file.write(as_string)


def get_feature_stat(features, bot_type=None):
	output_stat = {}
	for label in [f[1] for f in features[0:]]:
		if bot_type is not None:
			output_string = gi.as_to_str(gi.get_action_states(bot_type)[np.argmax(label)])
		else:
			output_string = gi.as_to_str(label)

		try:
			output_stat[output_string] += 1
		except KeyError:
			output_stat[output_string] = 1

	average_input = vmath.Vector(np.zeros([len(features[0][0])]).tolist())
	for vec_a in [f[0] for f in features[0:]]:
		average_input = average_input + vmath.Vector(vec_a)
	average_input = average_input.scalar_mul(1/len(features))

	similarity_info = DistributionInfo([calc_similarity(vec, average_input.values)[0] for vec in [f[0] for f in features[0:]]])
	angle_info = DistributionInfo([calc_similarity(vec, average_input.values)[1] for vec in [f[0] for f in features[0:]]])
	length_delta_info = DistributionInfo([calc_similarity(vec, average_input.values)[2] for vec in [f[0] for f in features[0:]]])

	return output_stat, average_input, similarity_info, angle_info, length_delta_info


def calc_similarity(a, b):
	if len(a) != len(b):
		print("in_sim - unequal length")
		return 0

	vec_a = vmath.Vector(a)
	vec_b = vmath.Vector(b)

	angle = vmath.angle(vec_a, vec_b)
	length_delta = abs(abs(vec_a) - abs(vec_b))
	similarity = ((180 - angle)/180) * (1/length_delta)

	return similarity, angle, length_delta


def nn_to_rlbot_controls(nn_controls):
	controller_state = SimpleControllerState()
	# steering
	if nn_controls[0] == 1:
		controller_state.steer = -1.0
	elif nn_controls[1] == 1:
		controller_state.steer = 1.0
	else:
		controller_state.steer = 0.0

	# pitch
	if nn_controls[2] == 1:
		controller_state.pitch = -1.0
	elif nn_controls[3] == 1:
		controller_state.pitch = 1.0
	else:
		controller_state.pitch = 0.0

	# throttle
	if nn_controls[4] == 1:
		controller_state.throttle = 1.0
	elif nn_controls[5] == 1:
		controller_state.throttle = -1.0
	else:
		controller_state.throttle = 0.0

		controller_state.jump = nn_controls[6] == 1
		controller_state.boost = nn_controls[7] == 1
		controller_state.handbrake = nn_controls[8] == 1

	if controller_state.handbrake:
		controller_state.roll = controller_state.steer
		controller_state.yaw = 0.0
	else:
		controller_state.roll = 0.0
		controller_state.yaw = controller_state.steer

	return controller_state


def rlbot_to_nn_controls(rlbot_controls):
	if rlbot_controls[0] < 0:
		acc = 0
		decc = 1
	elif rlbot_controls[0] > 0:
		acc = 1
		decc = 0
	else:
		acc = 0
		decc = 0

	if rlbot_controls[1] < 0:
		left = 1
		right = 0
	elif rlbot_controls[1] > 0:
		left = 0
		right = 1
	else:
		left = 0
		right = 0

	if rlbot_controls[2] < 0:
		down = 0
		up = 1
	elif rlbot_controls[2] > 0:
		down = 1
		up = 0
	else:
		down = 0
		up = 0

	jump = boost = slide = 0
	if rlbot_controls[5]:
		jump = 1
	if rlbot_controls[6]:
		boost = 1
	if rlbot_controls[7]:
		slide = 1

	return [left, right, up, down, acc, decc, jump, boost, slide]


def xbox_to_rlbot_controls(xbox_controls):
	# ----------steer----------
	# l_thumb_x
	if xbox_controls[0] == 128:
		steer = 0
	elif xbox_controls[0] < 128:
		steer = -1.0
	else:
		steer = 1.0

	# ----------pitch----------
	# l_thumb_y
	if xbox_controls[1] == 128:
		pitch = 0.0
	elif xbox_controls[1] < 128:
		pitch = 1.0
	else:
		pitch = -1.0

	# ----------throttle----------
	# right_trigger (acc)
	if xbox_controls[2] <= 10:
		xbox_controls[2] = 0
	else:
		xbox_controls[2] = 1
	# left_trigger (decc)
	if xbox_controls[3] <= 10:
		xbox_controls[3] = 0
	else:
		xbox_controls[3] = 1

	throttle = 0
	if xbox_controls[2] > 0:
		throttle = 1.0
	elif xbox_controls[3] > 0 and throttle < 1.0:
		throttle = -1.0
	else:
		throttle = 0.0

	# ---------jump----------
	if xbox_controls[5] > 0:
		jump = True
	else:
		jump = False

	# ---------boost----------
	if xbox_controls[4] > 0:
		boost = True
	else:
		boost = False

	# ---------slide----------
	if xbox_controls[6] > 0:
		slide = True
	else:
		slide = False

	# ---------yaw/roll---------
	if slide:
		roll = steer
		yaw = 0.0
	else:
		yaw = steer
		roll = 0.0

	return [throttle, steer, pitch, yaw, roll, jump, boost, slide]


def xbox_to_nn_controls(xbox_controls):
	# ----------steer----------
	if xbox_controls[0] == 128:
		left = right = 0
	elif xbox_controls[0] < 128:
		left = 0
		right = 1
	else:
		left = 1
		right = 0

	# ----------pitch----------
	if xbox_controls[1] == 128:
		up = down = 0
	elif xbox_controls[1] < 128:
		down = 1
		up = 0
	else:
		down = 0
		up = 1

	# ----------throttle----------
	# right_trigger (acc)
	if xbox_controls[2] <= 10:
		xbox_controls[2] = 0
	else:
		xbox_controls[2] = 1
	# left_trigger (decc)
	if xbox_controls[3] <= 10:
		xbox_controls[3] = 0
	else:
		xbox_controls[3] = 1

	acc = decc = 0
	if xbox_controls[2] > 0:
		acc = 1
	if xbox_controls[3] > 0:
		decc = 1

	# ---------jump----------
	if xbox_controls[5] > 0:
		jump = 1
	else:
		jump = 0

	# ---------boost----------
	if xbox_controls[4] > 0:
		boost = 1
	else:
		boost = 0

	# ---------slide----------
	if xbox_controls[6] > 0:
		slide = 1
	else:
		slide = 0

	return [right, left, up, down, acc, decc, jump, boost, slide]


def convert_to_float_list(string):
	string_list = string.split()
	float_list = []
	for s in string_list:
		try:
			float_list.append(float(s))
		except ValueError:
			continue
	return float_list


def convert_to_feature_string(feature):
	x = str(feature[0])
	y = str(feature[1])

	# take out unwanted chars
	x = x.replace('[', '').replace(']', '').replace('\'', '').replace(',', "").replace('\n', '').replace('-0.0', '0').replace('0.0', '0')
	y = y.replace('[', '').replace(']', '').replace('\'', '').replace(',', "").replace('\n', '').replace('-0.0', '0').replace('0.0', '0')

	feature_string = x + '::::' + y + '\n'
	return feature_string


def cull_input_log(src_path):
	non_noop_inputs = []
	count = 0
	with open(src_path, "r") as in_file:
		input_string = in_file.readline().rstrip("\n")
		while input_string != "":
			while count < 10000:
				if input_string != "000000000":
					non_noop_inputs.append(input_string)
					count += 1
				input_string = in_file.readline().rstrip("\n")

			print("next batch")
			with open(src_path, "a") as out_file:
				for input_string in non_noop_inputs:
					out_file.write(input_string + "\n")
				non_noop_inputs = []
				count = 0


def catalog_input(src_path):
	catalog = {}

	count = 0
	with open(src_path, "r") as src_file:
		in_str = src_file.readline().rstrip("\n")
		while in_str != "":
			try:
				catalog[in_str] += 1
			except KeyError:
				catalog[in_str] = 0
			count += 1
			if count % 100000 == 0:
				print(count/100000)

			in_str = src_file.readline().rstrip("\n")

	return catalog


if __name__ == '__main__':
	training_data, _ = read_all_v3("../log/training_data.txt", 7, 8)

	print("len td:", len(training_data))
	print("len tf[0]", len(training_data[0]))

	net = NeuralNetwork(7, 8, "test")
	net.fully_connected_layer(512)
	net.activation("relu")
	net.dropout(0.2)
	net.fully_connected_layer(512)
	net.activation("relu")
	net.commit()
	net.add_train_setup(learning_rate=0.1)

	net.train(training_data, batch_size=500, do_save=False)

	print("done")
