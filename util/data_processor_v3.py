from Networks.nn_provider import NeuralNetwork
from rlbot.agents.base_agent import SimpleControllerState


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

	controller_state.jump = (nn_controls[6] == 1)
	controller_state.boost = (nn_controls[7] == 1)
	controller_state.handbrake = (nn_controls[8] == 1)

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


def array_to_scs(a):
	scs = SimpleControllerState()

	scs.throttle = a[0]
	scs.steer = a[1]
	scs.pitch = a[2]
	scs.yaw = a[3]
	scs.roll = a[4]
	scs.jump = a[5]
	scs.boost = a[6]
	scs.handbrake = a[7]

	return scs


def scs_to_array(scs):
	# format: [throttle:float	(-1.0 ... 1.0)	neg:backwards, pos:forward
	# 			steer:float		(-1.0 ... 1.0)	neg:left, pos:right
	# 			pitch			(-1.0 ... 1.0)	neg:down, pos:up
	# 			yaw				(-1.0 ... 1.0)	neg:left, pos:right
	# 			roll			(-1.0 ... 1.0)	neg:left, pos:right
	# 			jump:boolean			TRUE:held
	# 			boost:boolean			TRUE:held
	# 			powerslide:boolean		TRUE:held
	# ]
	return [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]


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
