import util.game_info as gi
import math
from util.data_processor_v3 import scs_to_array, array_to_scs
from Networks.legacy.data_processor_v3 import nn_to_rlbot_controls, rlbot_to_nn_controls, xbox_to_nn_controls
from Networks.legacy.XInputReader import get_xbox_output as get_controller_output
from Networks.legacy.data_processor_v3 import get_feature_stat as data_stat
from util.stats import DistributionInfo as DistInfo
from Agents.ATBA.atba import ATBAgent
from Networks.legacy.nn_provider import NeuralNetwork
from time import time
from rlbot.agents.base_agent import BaseAgent

LOAD_NET = False
SAVE_NET = False
TRAIN = True
INFO = True
SPAM_INFO = False
FITNESS_INFO = True
INFO_INTERVAL = 10.0		# in seconds
LOG_DIR = "./log/"
NET_PATH = "E:/Studium/5. Semester/Rocket League Bot/RLBot_v3/RLBot/Trained_NNs/"
# LOG_TRAINING_DATA = True	# disabled

LOAD_NET_NAME = "FlowBot1522424200"
NET_NAME = "FlowBot" + str(math.floor(time()))
BOT_TYPE = "grounded"
N_INPUT = 1								# todo automate from INPUT_COMPOSITION
N_OUTPUT = len(gi.get_action_states(BOT_TYPE))
INDIVIDUALS_PER_GENERATION = 500
ITERATIONS_PER_INDIVIDUAL = 10			# default actions per second: 60 (set by the dll)
MAX_OUTPUT_REPETITIONS = 20				# how often the net-output can be repeated before the trainer takes over (does not count forced repetitions)
MIN_OUTPUT_REPETITIONS = 0				# how often outputs are automatically repeated
REPETITION_DECAY = 5					# how much the output repetition meter gets reduced after a change in output (set to equal MAX_OUTPUT_REPETITIONS to reduce to 0)
MIN_FITNESS = 1.0						# as a percentage of average fitness -> all individuals with fitness below MIN_FITNESS*100 % of the average will be culled
TRAINING_DATA_RETENTION = "alive"		# "all": retains all data, "alive": retains data of living individuals, "none"

TEACHER_ENABLE = True
TEACHING_STRATEGY = "demonstrate"		# "demonstrate": teacher does MAX_OUTPUT_REPETITIONS*TEACHER_STRENGTH actions in a row; "help": teacher does single action
TEACHER_STRENGTH = 1.5					# ratio between agent and teacher actions when demonstrating (higher -> teacher acts more)

ENABLE_USER_INPUT = True				# allows/disallows user input
TOGGLE_TEACHER = [0, 0, 0, 0, 1, 1, 0, 0, 0]		# if this user input is received the teacher is enabled/disabled

# how strongly the individual components of fitness contribute to the whole
FITNESS_COMPOSITION = {
	"angle_to_ball": 1.0,				# smaller angle is better
	"dist_from_ball": 0.0,				# less distance is better
	"speed": 0.0,						# faster is better
	"boost": 0.0,						# more boost is better
	"super_sonic": 0.0,					# being super sonic (95% of max speed) is better
	"increase": 5.0,					# improving the situation is better
	"action_resemblance": 0.0			# the closer the chosen actions resemble the actions taken by a human the better
}
MOMENT_FITNESS_PARTS = ["angle_to_ball", "dist_from_ball", "speed", "boost", "super_sonic"]									# list of all components that contribute to moment fitness
MAX_FITNESS = sum([100 * ITERATIONS_PER_INDIVIDUAL * FITNESS_COMPOSITION[key] for key in FITNESS_COMPOSITION])				# absolute maximum individual fitness
MAX_MOMENT_FITNESS = sum([100 * ITERATIONS_PER_INDIVIDUAL * FITNESS_COMPOSITION[key] for key in MOMENT_FITNESS_PARTS])		# absolute maximum moment fitness

# todo implement
INPUT_COMPOSITION = {

}


class FlowBot(BaseAgent):
	def __init__(self, name, team, index):
		super().__init__(name, team, index)
		self.ready = not LOAD_NET		# prevents the net from being called before its fully loaded

		if not TRAIN:
			print("\n-----WARNING: NOT TRAINING-----\n")

		self.name = name
		self.team = team   # 0 towards positive goal, 1 towards negative goal
		self.index = index

		self.prev_time = time()					# used to keep track of time since last info
		self.aps = 0							# actions per second
		self.agent_action_count = 0				# number of net-chosen actions
		self.teacher_action_count = 0			# number of teacher-chosen actions
		self.user_action_count = 0				# number of user-chosen actions
		self.prev_output = []					# action that was returned on most recent call
		self.prev_agent_action = []				# action the net chose on most recent call
		self.agent_output_repetitions = 0		# number of repeated actions chosen by the net
		self.output_cycle = 0					# keeps track of how often actions have been automatically repeated
		self.agent_output_stat = {}				# keeps track of all actions chosen by the net
		self.teacher_output_stat = {}			# keeps track of all actions chosen by the teacher
		self.teaching = False					# if true the teacher is given control; used for teaching in "demonstrate"-mode

		self.action_states = gi.get_action_states(BOT_TYPE)		# all actions the agent can choose from

		self.gen = 0					# number of completed generations
		self.iteration_count = 0		# number of completed iterations; is reset every time a generation is completed
		self.gen_data = []				# training data collected in the current generation
		self.cur_individual_data = {"game_info": [], "actions": [], "nn_input": [], "labels": []}		# training data collected in the current individual
		self.fitness_data = {}			# keeps track of all values associated with fitness

		self.teacher_enabled = TEACHER_ENABLE		# enables or disables the teacher completely; can be toggled through user input
		self.last_teacher_toggle = time()
		self.teacher = ATBAgent(name, team, index)
		# self.teacher = RandomAgent(name, team, index)

		if LOAD_NET:
			self.net = NeuralNetwork.restore(NET_PATH, LOAD_NET_NAME, new_id=NET_NAME)
		else:
			# builds a new net
			self.net = NeuralNetwork(N_INPUT, N_OUTPUT, NET_NAME, do_act_summaries=True, save_path=NET_PATH)
			self.net.fully_connected_layer(256)
			self.net.activation("relu")
			self.net.commit()
			self.net.add_train_setup(learning_rate=0.1)

		if INFO:
			print("NetFormat:", self.net.format)
			print("Fitness Composition:", FITNESS_COMPOSITION)
			print(INDIVIDUALS_PER_GENERATION, "Individuals with", ITERATIONS_PER_INDIVIDUAL, "Iterations")
			print("Teacher:", self.teacher, "(" + TEACHING_STRATEGY + ", " + str(TEACHER_STRENGTH) + ")")
			print("Data Retention:", TRAINING_DATA_RETENTION)

		# the net is ready to be called
		self.ready = True
		print("FlowBot ready\n")

	def get_output(self, game_tick_packet):
		"""
		this method is called by the BotManager and returns an action in RLBot-Controls-Format
		game_tick_packet is a struct containing all information collected by the RLBotCore.dll from the game
		:param game_tick_packet: contains the current state of the game
		:return: button presses the Agent will do in the next iteration (for format see end of method)
		"""
		self.aps += 1
		self.iteration_count += 1
		# if REPETITION_DECAY is greater than 1 it is possible for agent_output_repetitions to be less than 0
		if self.agent_output_repetitions < 0:
			self.agent_output_repetitions = 0

		# prepare the input vector with the data provided
		# the game_tick_packet struct is converted to a GameInfo-Object
		game_info = gi.GameInfo(game_tick_packet)
		# this_player = game_info.get_player(self.index)
		# a list of float values to be fed to the net
		nn_input_vector = [game_info.angle_to_ball()[1]]		# + game_info.as_array()		# [game_info.angle_to_ball()[1]] + game_info.ball_info.Location.as_list() # + this_player.Location.as_list()

		# calculate actions (agent and teacher)
		teacher_action = rlbot_to_nn_controls(scs_to_array(self.teacher.get_output(game_tick_packet)))
		agent_action = None
		if self.ready and (self.gen > 0 or LOAD_NET or not self.teacher_enabled):
			chosen_class, _ = self.net.predict(nn_input_vector, verbose=False)
			agent_action = self.action_states[chosen_class]

		# get the input from the controller
		user_action = xbox_to_nn_controls(get_controller_output())
		# enables/disables the teacher; can only be done once per second to prevent excessive toggling
		if user_action == TOGGLE_TEACHER and time() - self.last_teacher_toggle > 1:
			self.teacher_enabled = not self.teacher_enabled
			self.last_teacher_toggle = time()
			print("Teacher " + ("enabled" if self.teacher_enabled else "disabled"))
			user_action = None
		else:
			# tests if the user action is a valid action for the bot type; if not it is ignored
			try:
				gi.get_action_index(user_action, BOT_TYPE)
			except ValueError:
				user_action = None

		# choose action
		teacher_gen = (not LOAD_NET and self.gen == 0)		# its the first generation and no net was loaded
		output_override = self.agent_output_repetitions > MAX_OUTPUT_REPETITIONS		# the net exceeded its output repetition maximum
		self.teaching = self.teaching or output_override
		agent_acted = agent_action is not None		# no action was chosen by the net

		# user action; is chosen when user input is enabled an a valid action was read from the controller
		if ENABLE_USER_INPUT and user_action is not None:
			selected_action = user_action
			self.user_action_count += 1
			# print("user action:", user_action)
			actor = "user"
		# automatically repeated action; is chosen when the specified number of automatic repetitions has not been completed; is not chosen if no previous action was recorded
		elif self.output_cycle < MIN_OUTPUT_REPETITIONS and self.prev_output != []:
			selected_action = self.prev_output
			self.output_cycle += 1
			actor = "rep"
		# teacher action; is chosen if the no net action is available or the net action is overridden
		elif True or self.teacher_enabled and (not self.ready or not agent_acted or teacher_gen or output_override or self.teaching):		# todo remove True
			selected_action = teacher_action
			self.teacher_action_count += 1
			self.output_cycle = 0

			if self.teaching and TEACHING_STRATEGY == "demonstrate":
				self.agent_output_repetitions = max(float(0), self.agent_output_repetitions - 1/TEACHER_STRENGTH)
				self.teaching = not self.agent_output_repetitions <= 0
			else:
				self.agent_output_repetitions = max(0, self.agent_output_repetitions - REPETITION_DECAY)
				self.teaching = False
			actor = "teacher"
		# net action; is chosen when it is not overridden for any reason
		else:
			selected_action = agent_action
			self.agent_action_count += 1
			self.output_cycle = 0
			actor = "agent"

		# if the selected action contains a jump the action should not be auto repeated
		if selected_action[6] == 1:
			self.output_cycle = MIN_OUTPUT_REPETITIONS

		# counting the nets action repetitions and reduce the count if the net chooses different actions
		if actor == "agent":
			if agent_action == self.prev_agent_action:
				self.agent_output_repetitions += 1
			else:
				self.agent_output_repetitions = max(0, self.agent_output_repetitions-REPETITION_DECAY)

		# additional info given every iteration
		if SPAM_INFO and self.gen > 0:
			print("Actor:", actor)
			print("Repetitions:", self.agent_output_repetitions)
			print()

		# convert the chosen action to RLBot-Control-Format
		return_vector = nn_to_rlbot_controls(selected_action)

		# ------------------------------------------------------------------------------------------------------------------

		# record training data for later processing
		# label is a one-hot vector describing which action was chosen
		label = [0 for _ in range(len(self.action_states))]
		label[gi.get_action_index(selected_action, BOT_TYPE)] = 1

		self.cur_individual_data["game_info"].append(game_info)
		self.cur_individual_data["actions"].append(selected_action)
		self.cur_individual_data["nn_input"].append(nn_input_vector)
		self.cur_individual_data["labels"].append(label)

		# an individual is completed
		if self.iteration_count % ITERATIONS_PER_INDIVIDUAL == 0:
			self.gen_data.append(self.cur_individual_data)
			self.cur_individual_data = {"game_info": [], "actions": [], "nn_input": [], "labels": []}

		# previous info gets updated for the following iteration
		self.prev_output = selected_action
		self.prev_agent_action = agent_action

		# record the output for analysis
		try:
			self.teacher_output_stat[gi.as_to_str(teacher_action)] += 1
		except KeyError:
			self.teacher_output_stat[gi.as_to_str(teacher_action)] = 1

		if agent_action is not None:
			try:
				self.agent_output_stat[gi.as_to_str(agent_action)] += 1
			except KeyError:
				self.agent_output_stat[gi.as_to_str(agent_action)] = 1

		# when a generation is completed the collected data is processed and the net is (re-)trained
		if self.iteration_count >= INDIVIDUALS_PER_GENERATION * ITERATIONS_PER_INDIVIDUAL:
			self.iteration_count = 0
			print("")
			# print("generation,", self.gen, "completed")
			print("Agent Action Count:", self.agent_action_count)
			print("Agent Output Statistic", self.agent_output_stat)
			print("Teacher Action Count:", self.teacher_action_count)
			print("Teacher Output Statistic", self.teacher_output_stat)
			print("User Action Count:", self.user_action_count)
			print("")

			# training the net with the acquired data
			if TRAIN:
				print("net training...")
				self.net.train(self.prepare_training_data(), batch_size=1000, do_save=True, n_epochs=2, auto_load=LOAD_NET)

			# reset counts and stats for the next generation
			self.agent_action_count = 0
			self.teacher_action_count = 0
			self.user_action_count = 0
			for key in self.agent_output_stat:
				self.agent_output_stat[key] = 0
			for key in self.teacher_output_stat:
				self.teacher_output_stat[key] = 0

			self.gen += 1
			print("\n------------------------Generation " + str(self.gen) + "------------------------")

		# info for debugging purposes
		cur_time = time()
		if INFO and cur_time-self.prev_time > INFO_INTERVAL:
			# print("\n------------------------Info------------------------")
			print("Progress:", str(round((self.iteration_count / (INDIVIDUALS_PER_GENERATION * ITERATIONS_PER_INDIVIDUAL)) * 100, 2)) + "% (" + str(self.aps/INFO_INTERVAL) + "a/s)")
			print(self.team)
			print("Running", self.aps/INFO_INTERVAL, "a/s")
			print("Input Vector:", nn_input_vector)
			print("Selected Action:", selected_action)
			print("Agent Action Count:", self.agent_action_count)
			print("Agent Output Statistic", self.agent_output_stat)
			print("Teacher Action Count:", self.teacher_action_count)
			print("Teacher Output Statistic", self.teacher_output_stat)
			print("Agent Output Repetitions:", self.agent_output_repetitions)
			print("Label:", label)
			print("Return Vector:", return_vector)
			print("------------------------------------------------------")
			# print("\n")

			self.prev_time = cur_time		# set time of previous info to now
			self.aps = 0					# reset actions per second

		# format: [throttle:float	(-1.0 ... 1.0)	neg:backwards, pos:forward
		# 			steer:float		(-1.0 ... 1.0)	neg:left, pos:right
		# 			pitch			(-1.0 ... 1.0)	neg:down, pos:up
		# 			yaw				(-1.0 ... 1.0)	neg:left, pos:right
		# 			roll			(-1.0 ... 1.0)	neg:left, pos:right
		# 			jump:boolean			TRUE:held
		# 			boost:boolean			TRUE:held
		# 			powerslide:boolean		TRUE:held
		# ]
		return array_to_scs(return_vector)

	def prepare_training_data(self, min_fitness=MIN_FITNESS):
		"""
		compiles and processes the collected data so it can be fed to the net for training
		min_fitness is the minimum fitness an individual has to have to survive; given in % of the average fitness
		:param min_fitness: the minimum fitness required for an individual to survive
		:return: the training data ready to be feed to the net
		"""
		# calculate the fitness for all individuals
		for i in self.gen_data:
			i["fitness"] = self.individual_fitness(i)
		# calculate average fitness of all individuals; negative average fitness is set to 0
		avrg_fitness = max(0, sum([i["fitness"] for i in self.gen_data]) / len(self.gen_data))

		killed = 0				# number of killed individuals
		training_data = []		# processed training data to be fed to the net
		survivors = []			# raw data of the surviving individuals
		avrg_deviation = 0		# average deviation from the average fitness
		for individual in self.gen_data:
			avrg_deviation += abs(1 - individual["fitness"]/avrg_fitness)
			if individual["fitness"]/avrg_fitness > min_fitness:
				# add all iterations of an individual to the processed data
				for i in range(ITERATIONS_PER_INDIVIDUAL):
					training_data.append([individual["nn_input"][i], individual["labels"][i]])
				survivors.append(individual)
			else:
				killed += 1

		avrg_deviation /= len(self.gen_data)

		# general information about the generation
		print("Total Average Fitness:", avrg_fitness)
		print("Survivors Average Fitness:", sum([i["fitness"] for i in survivors]) / len(survivors))
		print("Average Deviation:", avrg_deviation)
		print("Killed", killed, "Individual(s)")
		print("Survivors:", len(survivors))

		print("Data Statistic:")
		# calculate statistics for the processed training data
		output_stat, average_input, similarity_info, angle_info, length_delta_info = data_stat(training_data, bot_type=BOT_TYPE)
		print("\tOutput Statistic:", output_stat)						# what actions are contained in the data
		print("\tAverage Input:", average_input)						# elementwise average of input values
		print("\tInput Similarity", str(similarity_info))				# value representing how similar the input tensors are to each other
		print("\tAngle Component", str(angle_info))						# part of the input similarity value
		print("\tLength Delta Component", str(length_delta_info))		# part of the input similarity value

		# additional information about the fitness values in this generation
		if FITNESS_INFO:
			print("Fitness Statistic (all):")
			for key in self.fitness_data:
				print("\t" + key + ":", DistInfo(self.fitness_data[key]))
		self.fitness_data = {}

		print()		# console output formatting

		# retain the specified data into the next generation
		if TRAINING_DATA_RETENTION == "none":
			self.gen_data = []		# no data is retained
		elif TRAINING_DATA_RETENTION == "alive":
			self.gen_data = survivors		# data from surviving individuals is retained
		# if TRAINING_DATA_RETENTION is not none and not alive all data is retained

		return training_data

	def individual_fitness(self, individual_data):
		"""
		calculates the fitness value of a single individual
		:param individual_data: all data collected during the run of an idividual
		:return: the fitness value the individual achieved
		"""
		fitness = 0
		ind_fit_dict = {}		# used for additional information during runtime

		# add the sum of moment fitness values from each iteration
		for game_info in individual_data["game_info"]:
			fitness += self.moment_fitness(game_info, ind_fit_dict)

		# fitness from increase/decrease of moment fitness from beginning to end of the individual
		starting_fitness = self.moment_fitness(individual_data["game_info"][0])
		ending_fitness = self.moment_fitness(individual_data["game_info"][-1])
		increase = ITERATIONS_PER_INDIVIDUAL * 100 * (ending_fitness - starting_fitness) * FITNESS_COMPOSITION["increase"]
		fitness += increase

		# fitness from how similar the actions are to the expected actions (based on human play)
		composition = 0
		if FITNESS_COMPOSITION["action_resemblance"] != 0:
			# record the actions in the data
			actions = {}
			for action in individual_data["actions"]:
				try:
					actions[gi.as_to_str(action)] += 1
				except KeyError:
					actions[gi.as_to_str(action)] = 1

			# the distribution of actions taken by human players
			bot_action_percentages = gi.get_action_percentages(BOT_TYPE, normalize=True)
			deviation_sum = 0
			max_deviation_sum = 1 / min([bot_action_percentages[key] for key in bot_action_percentages])
			for key in actions:
				# |1 - actions_taken/actions_expected|
				deviation_sum += abs(1 - (actions[key] / ITERATIONS_PER_INDIVIDUAL) / bot_action_percentages[key])

			composition = ITERATIONS_PER_INDIVIDUAL * 100 * (1 - deviation_sum / max_deviation_sum) * FITNESS_COMPOSITION["action_resemblance"]
			fitness += composition

		# additional info about fitness
		if FITNESS_INFO:
			try:
				ind_fit_dict["increase"] += increase
			except KeyError:
				ind_fit_dict["increase"] = increase
			try:
				ind_fit_dict["composition"] += composition
			except KeyError:
				ind_fit_dict["composition"] = composition

			for key in ind_fit_dict:
				try:
					self.fitness_data[key].append(ind_fit_dict[key])
				except KeyError:
					self.fitness_data[key] = [ind_fit_dict[key]]

		return fitness

	def moment_fitness(self, game_info, fitness_dict=None):
		"""
		calculates the fitness for a given situation(GameInfo-Object)
		:param game_info: the state of the game
		:param fitness_dict: dictionary holding stats about fitness (only needed for analysis)
		:return: the fitness earned for beeing in the state
		"""
		own_car = game_info.get_player(self.index)

		angle = 100 * (abs(game_info.angle_to_ball(0)[1])/180) * FITNESS_COMPOSITION["angle_to_ball"]
		distance = 100 * ((13175 - game_info.dist_to_ball(0)[1]) / 13175) * FITNESS_COMPOSITION["dist_from_ball"]
		speed = 100 * (abs(own_car.velocity) / gi.MAX_VELOCITY) * FITNESS_COMPOSITION["speed"]
		boost = own_car.boost * FITNESS_COMPOSITION["boost"]
		super_sonic = (100 if own_car.is_super_sonic else 0) * FITNESS_COMPOSITION["super_sonic"]

		if FITNESS_INFO and fitness_dict is not None:
			try:
				fitness_dict["angle_to_ball"] += angle
			except KeyError:
				fitness_dict["angle_to_ball"] = angle
			try:
				fitness_dict["dist_from_ball"] += distance
			except KeyError:
				fitness_dict["dist_from_ball"] = distance
			try:
				fitness_dict["speed"] += speed
			except KeyError:
				fitness_dict["speed"] = speed
			try:
				fitness_dict["boost"] += boost
			except KeyError:
				fitness_dict["boost"] = boost
			try:
				fitness_dict["super_sonic"] += super_sonic
			except KeyError:
				fitness_dict["super_sonic"] = super_sonic

		return angle + distance + speed + boost + super_sonic

	def __str__(self):
		return "FB_" + NET_NAME + "(" + str(self.index) + ") " + ("blue" if self.team == 0 else "orange")


def log(content):
	"""
	writes a string to the specified logfile
	:param content: will be converted to string and written to the logfile
	"""

	with open(LOG_DIR + "default.txt", "a") as log_file:
		log_file.write(str(content) + "\n")
