import os
import random
import numpy as np
from time import time

import util.game_info as gi
from util.information import RunInfo
from util.data_processor_v3 import nn_to_rlbot_controls, xbox_to_nn_controls
from util.XInputReader import get_xbox_output as get_controller_output
from Networks.q_learning import NeuralNetwork, ActivationType, ReplayMemory

from rlbot.agents.base_agent import BaseAgent

# load/save behavior
TRAIN = True
SAVE_NET = True
COLLECT_DATA = True
SAVE_DATA = COLLECT_DATA and True
LOAD_NET = False
LOAD_NET_NAME = "FlowBot1530614290"
NET_PATH = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/Networks/saved/"
TEMP_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/temp/"
LOG_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/logs/"

# info
INFO = True
EPISODE_INFO = True
SPAM_INFO = False
FITNESS_INFO = True
INFO_INTERVAL = 10.0		# in seconds

# net and training properties
NET_NAME = "FlowBot" + str(int(time()))
BOT_TYPE = "flying"
N_INPUT = 29							# todo automate from INPUT_COMPOSITION
N_OUTPUT = len(gi.get_action_states(BOT_TYPE))
START_EPSILON = 0.9						# chance that a random action will be chosen instead of the one with highest q_value
EPSILON_DECAY = 1e-3					# amount the epsilon value decreases every episode
MIN_EPSILON = 0.1						# minimum epsilon value

# how strongly the individual components of fitness contribute to the whole
STATE_SCORE_COMPOSITION = {
	"angle_to_ball": 0.0,				# smaller angle is better
	"dist_from_ball": 0.0,				# less distance is better
	"speed": 0.0,						# faster is better
	"boost": 0.0,						# more boost is better
	"super_sonic": 0.0					# being super sonic (95% of max speed) is better
}
MAX_STATE_SCORE = sum([STATE_SCORE_COMPOSITION[key] for key in STATE_SCORE_COMPOSITION])					# maximum score a state can have

# todo implement
INPUT_COMPOSITION = {

}

USER_INPUT_ENABLED = False				# allows/disallows user input
USER_OPTIONS = {"toggle_user_input": [0, 0, 0, 0, 1, 1, 0, 0, 0]}						# special inputs the user can make to change parameters of the bot


class FlowBot(BaseAgent):
	def __init__(self, name, team, index):
		super().__init__(name, team, index)
		if not TRAIN:
			print("\n-----NOT TRAINING-----\n")
		os.makedirs(LOG_DIR + NET_NAME)

		self.prev_info_time = time()				# used to keep track of time since last info
		self.action_states = gi.get_action_states(BOT_TYPE)		# all actions the agent can choose from

		self.episode_end_condition = EpisodeEndCondition(landed=True)
		self.epsilon = START_EPSILON
		self.aps = 0						# actions per second

		# list of tuples of (state, action, reward);
		self.replay_memory = ReplayMemory(n_actions=N_OUTPUT)
		self.run_info = RunInfo()  # holds all information
		self.prev_state = None
		self.prev_action = None
		self.prev_q_values = None
		self.prev_game_info = None

		self.user_input_cool_down = time()

		if LOAD_NET:
			self.net = NeuralNetwork.restore(LOAD_NET_NAME, NET_PATH, new_name=NET_NAME, verbose=True)
		else:
			self.net = NeuralNetwork(NET_NAME, [N_INPUT], N_OUTPUT)
			self.net.add_fc(512, activation=ActivationType.RELU)
			self.net.add_fc(512, activation=ActivationType.RELU)
			self.net.add_fc(512, activation=ActivationType.RELU)
			self.net.commit()
			self.run_info.net = self.net

		if INFO:
			print("FlowBot[name={0:s}], team={1:d}, index={2:d}".format(name, team, index))
			print("Fitness Composition:", STATE_SCORE_COMPOSITION)

		# the net is ready to be called
		self.net_ready = True
		print("FlowBot ready\n")

	def get_output(self, game_tick_packet):
		self.aps += 1

		game_info = gi.GameInfo(game_tick_packet)		# the game_tick_packet struct is converted to a GameInfo-Object
		state = game_info.full_state()					# a list of float values to be fed to the net

		# set the reward for the previous iteration; not possible in the first iteration because not previous state and action are available
		if self.prev_state is not None and self.prev_q_values is not None and self.prev_action is not None:
			self.replay_memory.add(state=self.prev_state,
								q_values=self.prev_q_values,
								action=self.prev_action,
								reward=self.reward(game_info))
		self.prev_state = state
		self.prev_game_info = game_info

		predicted_q_values = self.net.run([state])
		if random.random() < self.epsilon:
			chosen_class = random.randrange(0, len(predicted_q_values))
		else:
			chosen_class = np.argmax(predicted_q_values)
		agent_action = self.action_states[chosen_class]

		self.prev_q_values = predicted_q_values

		# get the input from the controller
		user_action = xbox_to_nn_controls(get_controller_output())
		# executes a special user action
		if time() - self.user_input_cool_down > 1:
			for key in USER_OPTIONS:
				if user_action == USER_OPTIONS[key]:
					if key == "toggle_user_input":
						global USER_INPUT_ENABLED
						USER_INPUT_ENABLED = not USER_INPUT_ENABLED
		if user_action not in self.action_states:
			user_action = None

		# user action; is chosen when user input is enabled and the action was valid
		if USER_INPUT_ENABLED and user_action is not None:
			selected_action = user_action
			actor = "user"
		# net action
		else:
			selected_action = agent_action
			actor = "agent"

		# convert the chosen action to RLBot-Control-Format
		return_controller_state = nn_to_rlbot_controls(selected_action)
		self.prev_action = gi.get_action_index(selected_action, BOT_TYPE)

		# at the end of every episode the data in the replay memory is updated and the net is trained with the new data
		if self.episode_end_condition.is_met(game_info) and TRAIN:
			self.next_episode()

		# info for debugging purposes
		cur_time = time()
		if INFO and cur_time-self.prev_info_time > INFO_INTERVAL:
			print("\n------------------------Info------------------------")
			print("Running", self.aps/INFO_INTERVAL, "a/s")
			# print("Game state:", game_info)
			print("Epsilon:", self.epsilon)
			print("Memory size:", self.replay_memory.size)
			print("Net input:", state)
			print("Net Output:", str(predicted_q_values))		# todo does NOT print predicted_q_values (only prints array of 0s)
			print("Action:", selected_action)
			# print("Return Vector:", return_controller_state)
			# print("------------------------------------------------------")
			print()

			self.prev_info_time = cur_time		# set time of previous info to now
			self.aps = 0					# reset actions per second

		self.run_info.iteration(net_output=predicted_q_values, action=selected_action, user=(actor == "user"), verbose=SPAM_INFO)

		return return_controller_state

	# todo improve
	def reward(self, cur_game_info):
		"""
		calculates the reward the agent receives for transitioning from state to new_state using the chosen action
		:param cur_game_info: the state the agent moved to
		:return: the reward for the change in state
		"""

		reward = cur_game_info.get_player(self.index).location.z / gi.CEILING_HEIGHT
		# reward = self.state_score(cur_game_info) - self.state_score(self.prev_game_info)
		return reward

	def next_episode(self):
		"""
		updates the q_values in the replay memory and trains the net
		:return: the time it took to 1: update the qvs and 2: train the net
		"""
		# update the q_values in the replay memory
		mem_up_time = self.replay_memory.update_q_values()

		# decrease epsilon
		self.epsilon = round(self.epsilon - EPSILON_DECAY, 5)
		if self.epsilon < MIN_EPSILON:
			self.epsilon = MIN_EPSILON

		# todo vary learning rate
		# train the net
		train_start = time()
		self.net.train(self.replay_memory, batch_size=512, n_epochs=4, save=SAVE_NET)
		train_end = time()

		self.run_info.episode(mem_up_time, train_end - train_start, verbose=EPISODE_INFO)
		# write data to file
		if COLLECT_DATA:
			self.replay_memory.write()
			self.run_info.write()

		# todo research different strategies for replay memory
		self.replay_memory.clear()

	def state_score(self, game_info):
		"""
		calculates the fitness for a single state(GameInfo-Object)
		:param game_info: the GameInfo-object that contains the state information
		:return: the score of the state (integer >=0)
		"""
		own_car = game_info.get_player(self.index)

		angle = (abs(game_info.angle_to_ball(0)[1])/180) * STATE_SCORE_COMPOSITION["angle_to_ball"]
		distance = ((13175 - game_info.dist_to_ball(0)[1]) / 13175) * STATE_SCORE_COMPOSITION["dist_from_ball"]
		speed = (abs(own_car.velocity) / gi.MAX_VELOCITY) * STATE_SCORE_COMPOSITION["speed"]
		boost = 0.01 * own_car.boost * STATE_SCORE_COMPOSITION["boost"]
		super_sonic = (1 if own_car.is_super_sonic else 0) * STATE_SCORE_COMPOSITION["super_sonic"]

		state_score = (angle + distance + speed + boost + super_sonic) / MAX_STATE_SCORE
		self.run_info.state_score(angle, distance, speed, boost, super_sonic)
		return state_score

	def retire(self):
		print("retire")
		if SAVE_NET:
			self.net.save()
		self.net.close()
		if SAVE_DATA:
			for _, _, files in os.walk(TEMP_DIR):
				for file in files:
					with open(TEMP_DIR + file, "r") as src, open(LOG_DIR + NET_NAME + "/" + file, "w") as dest:
						dest.write(src.read())
					with open(TEMP_DIR + file, "w") as tmp:
						tmp.write("")

	def __str__(self):
		return "FBv2_" + NET_NAME + "(" + str(self.index) + ") " + ("blue" if self.team == 0 else "orange")


class EpisodeEndCondition:
	def __init__(self, fixed_length=None, goal=False, game_end=False, landed=False):
		self.fl = fixed_length
		self.gs = goal
		self.ge = game_end
		self.la = landed

		self.iteration_count = 0
		self.was_round_active = True
		self.was_on_ground = True

		self.was_met_fl = False
		self.was_met_gs = False
		self.was_met_ge = False
		self.was_met_la = False

	def is_met(self, state, player_id=0):
		self.iteration_count += 1
		is_on_ground = state.get_player(player_id).is_on_ground

		fl_reached = self.fl is not None and self.iteration_count >= self.fl
		goal_scored = self.gs and not state.is_round_active and self.was_round_active
		game_ended = self.ge and state.is_match_ended
		has_landed = self.la and not self.was_on_ground and is_on_ground

		self.was_round_active = state.is_round_active
		self.was_on_ground = is_on_ground

		self.was_met_fl = fl_reached
		self.was_met_gs = goal_scored
		self.was_met_ge = game_ended
		self.was_met_la = has_landed

		is_met = fl_reached or goal_scored or game_ended or has_landed
		if is_met:
			self.iteration_count = 0

		return is_met
