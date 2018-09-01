import os
import random
import numpy as np
from time import time
from configparser import ConfigParser

import util.game_info as gi
from util.game_info import state_size
from util.vector_math import angle, Vector3
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
LOAD = False
PRESERVE = True
LOAD_BOT_NAME = "FlowBot1533812543"
PROJECT_ROOT = str(__file__).replace("Agents\\FlowBotv2\\flow_bot.py", "")
NET_PATH = PROJECT_ROOT + "Networks/saved/"
TEMP_DIR = PROJECT_ROOT + "util/temp/"
LOG_DIR = PROJECT_ROOT + "util/logs/"

# info
INFO = True
EPISODE_INFO = False
SPAM_INFO = False
RENDER = False
INFO_INTERVAL = 10.0					# in seconds
FULL_SAVE_INTERVAL = 5					# how often the bot along with all collected data is saved (in iterations)
SAVE_INTERVAL = 1						# how often the bot is saved (in iterations)


# net and training properties
NET_NAME = "FlowBot_tob" + str(int(time()))
BOT_TYPE = "all"
INPUT_COMPOSITION_FILE = PROJECT_ROOT + "Agents/FlowBotv2/state_composition.cfg"
N_OUTPUT = len(gi.get_action_states(BOT_TYPE))
START_EPSILON = 0.9						# chance that a random action will be chosen instead of the one with highest q_value
EPSILON_DECAY = 1e-3					# amount the epsilon value decreases every episode (default 5e-4)
EPSILON_STARTUP_DECAY = 0				# amount the epsilon value decreases every time the bot is loaded (not the first time)
MIN_EPSILON = 0.1						# minimum epsilon value
USE_SARSA = False
RELATIVE_COORDINATES = True
ALLOW_NEGATIVE_REWARD = False

# end conditions
EC_FIXED_LENGTH = 500
EC_GOAL = False
EC_GAME_END = False
EC_LANDED = False
END_CONDITIONS = [EC_FIXED_LENGTH, EC_GOAL, EC_GAME_END, EC_LANDED]

# rewards
RE_HEIGHT = False						# car.z / ceiling_height
RE_AIRTIME = False						# +1 for every iteration where !car.on_ground (given when landed)
RE_BALL_DIST = False					# distance between car and ball
RE_FACING_UP = False					# angle between z-axes and car.facing (normalized to 0-1)
RE_FACING_OPP = False					# angle between y-axes and car.facing (normalized to 0-1)
RE_FACING_BALL = True					# angle between car->ball and car.facing (normalized to 0-1)
REWARDS = [RE_HEIGHT, RE_AIRTIME, RE_BALL_DIST, RE_FACING_UP, RE_FACING_OPP, RE_FACING_BALL]
REWARD_EXP = 2

USER_INPUT_ENABLED = False				# allows/disallows user input
USER_OPTIONS = {"toggle_user_input": [0, 0, 0, 0, 1, 1, 0, 0, 0]}						# special inputs the user can make to change parameters of the bot


class FlowBot(BaseAgent):
	def __init__(self, name, team, index):
		super().__init__(name, team, index)

		if not TRAIN:
			print("\n-----NOT TRAINING-----\n")

		# clear the contents of temp info files and the log
		clear_temp()
		with open(LOG_DIR + "log.txt", "w") as f:
			f.write("")

		self.name = NET_NAME
		self.prev_info_time = time()				# used to keep track of time since last info
		self.action_states = gi.get_action_states(BOT_TYPE)		# all actions the agent can choose from

		self.state_comp = ConfigParser()
		self.state_comp.read(INPUT_COMPOSITION_FILE)

		self.episode_end_condition = EpisodeEndCondition()
		self.epsilon = START_EPSILON
		self.epsilon_decay = EPSILON_DECAY
		self.sarsa = USE_SARSA
		self.rel_coords = RELATIVE_COORDINATES
		self.reward_exp = REWARD_EXP
		self.aps = 0						# actions per second

		# list of tuples of (state, action, reward);
		self.replay_memory = ReplayMemory(n_actions=N_OUTPUT)
		self.run_info = RunInfo()  # holds all information
		self.prev_state = None
		self.prev_action = None
		self.prev_q_values = None
		self.prev_game_info = None
		self.reward_accumulator = 0		# for reward functions that need more than one iteration

		self.user_input_cool_down = time()

		if LOAD:
			self.load(preserve=PRESERVE)
		else:
			self.net = NeuralNetwork(NET_NAME, [state_size(self.state_comp)], len(self.action_states))
			self.net.add_fc(256, activation=ActivationType.RELU)
			self.net.add_drop_out(0.2)
			self.net.add_fc(512, activation=ActivationType.RELU)
			self.net.add_drop_out(0.2)
			self.net.add_fc(256, activation=ActivationType.RELU)
			self.net.add_drop_out(0.2)
			self.net.commit()
			self.run_info.net = self.net

		if INFO:
			print("FlowBot[name={0:s}], team={1:d}, index={2:d}".format(self.name, self.team, self.index))
			print("Fitness Composition:", REWARDS)

		# the net is ready to be called
		self.net_ready = True
		print("FlowBot ready\n")

	def get_output(self, game_tick_packet):
		self.aps += 1

		game_info = gi.GameInfo(game_tick_packet)		# the game_tick_packet struct is converted to a GameInfo-Object
		if self.team == gi.ORANGE_TEAM:
			game_info.mirror()

		if self.rel_coords:
			state = game_info.get_relative(self.index).get_state(self.index, self.state_comp)
		else:
			state = game_info.get_state(self.index, self.state_comp)		# a list of float values to be fed to the net

		# set the reward for the previous iteration; not possible in the first iteration because not previous state and action are available
		reward = 0
		if self.prev_state is not None and self.prev_q_values is not None and self.prev_action is not None:
			reward = self.reward(game_info)
			self.replay_memory.add(state=self.prev_state,
								q_values=self.prev_q_values,
								action=self.prev_action,
								reward=reward)
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
		user_action = None
		try:
			user_action = xbox_to_nn_controls(get_controller_output())
		except AttributeError as e:
			if SPAM_INFO:
				print("no controller found")

		# executes a special user action
		if time() - self.user_input_cool_down > 1 and user_action is not None:
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
		if self.episode_end_condition.is_met(game_info):
			self.next_episode()

			if self.run_info.episode_count % FULL_SAVE_INTERVAL == 0:
				self.save(info_files=True)
			elif self.run_info.episode_count % SAVE_INTERVAL == 0:
				self.save()

		# info for debugging purposes
		cur_time = time()
		if INFO and cur_time-self.prev_info_time > INFO_INTERVAL:
			print("\n------------------------Info------------------------")
			print("Running", self.aps/INFO_INTERVAL, "a/s")
			print("Episode", self.run_info.episode_count)
			# print("Game state:", game_info)
			print("Epsilon:", round(self.epsilon, 5))
			# print("Memory size:", self.replay_memory.size)
			# print("Net input:", state)
			# print("Net Output:", str(predicted_q_values))
			# print("Action:", selected_action)
			# print("Return Vector:", return_controller_state)
			# print("------------------------------------------------------")
			# print()

			self.prev_info_time = cur_time		# set time of previous info to now
			self.aps = 0					# reset actions per second

		self.run_info.iteration(net_output=predicted_q_values, action=selected_action, user=(actor == "user"), verbose=SPAM_INFO)

		if RENDER:
			self.render(gi.GameInfo(game_tick_packet), reward)

		return return_controller_state

	def next_episode(self):
		"""
		updates the q_values in the replay memory and trains the net
		:return: the time it took to 1: update the qvs and 2: train the net
		"""
		# decrease epsilon
		self.epsilon = self.epsilon - self.epsilon_decay
		if self.epsilon < MIN_EPSILON:
			self.epsilon = MIN_EPSILON

		mem_up_time, train_start, train_end = 0, 0, 0
		if TRAIN:
			# update the q_values in the replay memory
			mem_up_time = self.replay_memory.update_q_values(sarsa=self.sarsa)
			# todo vary learning rate
			# train the net
			train_start = time()
			self.net.train(self.replay_memory.get_training_set(), batch_size=512, n_epochs=4, save=SAVE_NET)
			train_end = time()

		self.run_info.episode(mem_up_time, train_end - train_start, verbose=EPISODE_INFO)
		# write data to file
		if COLLECT_DATA:
			self.replay_memory.write()
			self.run_info.write()

		# todo research different strategies for replay memory
		self.replay_memory.clear()

	def reward(self, cur_game_info):
		"""
		calculates the reward the agent receives for transitioning from one state(self.prev_game_info) to another(cur_game_info) using the chosen action
		:param cur_game_info: the state the agent moved to
		:return: the reward for the change in state
		"""
		car = cur_game_info.get_player(self.index)
		facing = Vector3.from_list(car.get_basis()[0])

		reward = 0
		h_component = 0
		at_component = 0
		bd_component = 0
		fu_component = 0
		fo_component = 0
		fb_component = 0

		if RE_AIRTIME:
			if self.episode_end_condition.was_met:
				at_component = self.reward_accumulator / self.run_info.episode_lengths[-1]
				reward += at_component
				self.reward_accumulator = 0
			else:
				if not car.is_on_ground:
					self.reward_accumulator += 1

		if RE_HEIGHT:
			h_component = (car.location.z - gi.FLOOR_HEIGHT) / gi.ARENA_HEIGHT
			reward += h_component

		if RE_FACING_UP:
			vertical = Vector3(0, 0, 1)
			fu_component = angle(facing, vertical) / 180
			fu_component = (abs(fu_component) - 0.5) * 2

			if not ALLOW_NEGATIVE_REWARD:
				fu_component = max(0, fu_component)

			reward += fu_component

		if RE_FACING_OPP:
			forward = Vector3(0, 1, 0)

			fo_component = angle(facing, forward) / 180
			fo_component = (abs(fo_component) - 0.5) * 2

			if not ALLOW_NEGATIVE_REWARD:
				fo_component = max(0, fo_component)

			reward += fo_component

		if RE_FACING_BALL:
			fb_component = (180 - cur_game_info.angle_to_ball(self.index)[0]) / 180
			fb_component = (abs(fb_component) - 0.5) * 2

			if not ALLOW_NEGATIVE_REWARD:
				fb_component = max(0, fb_component)

			reward += fb_component

		if RE_BALL_DIST:
			bd_component = cur_game_info.dist_to_ball(player_id=self.index)[0]
			if bd_component != gi.MAX_DIST:
				bd_component = max(0, (gi.MAX_DIST - bd_component) / gi.MAX_DIST)

			reward += bd_component

		self.run_info.reward(h_component, at_component, bd_component, fu_component, fo_component, fb_component)

		if not ALLOW_NEGATIVE_REWARD:
			reward = max(0, reward)
		n_enabled_rewards = len([b for b in REWARDS if b])
		reward /= n_enabled_rewards

		is_neg = reward < 0		# raising reward to an even power would result in loss of sign
		reward = abs(reward) ** self.reward_exp
		if ALLOW_NEGATIVE_REWARD and is_neg:
			return -reward
		else:
			return reward

	def render(self, state, reward=0):
		r = self.renderer

		car = state.get_player(self.index)
		ball = state.ball_info

		r.begin_rendering()

		# some default colors
		red = r.create_color(255, 255, 0, 0)
		green = r.create_color(255, 0, 255, 0)
		blue = r.create_color(255, 0, 0, 255)

		text_color = r.white()

		# info
		r.draw_string_2d(8, 16, 1, 1, "Episode: " + str(self.run_info.episode_count), text_color)
		r.draw_string_2d(8, 32, 1, 1, "Epsilon: " + str(self.epsilon), text_color)
		r.draw_string_2d(8, 48, 1, 1, "Reward: " + str("{0:.3f}").format(reward), text_color)

		# basis of the relative coordinates
		basis_x, basis_y, basis_z = car.get_basis(as_v3=True)
		basis_x = basis_x.normalize().scalar_mul(100)
		basis_y = basis_y.normalize().scalar_mul(100)
		basis_z = basis_z.normalize().scalar_mul(100)
		pos = car.location.as_list()
		x_line_end = (car.location + basis_x).as_list()
		r.draw_line_3d(pos, x_line_end, color=red)
		y_line_end = (car.location + basis_y).as_list()
		r.draw_line_3d(pos, y_line_end, color=blue)
		z_line_end = (car.location + basis_z).as_list()
		r.draw_line_3d(pos, z_line_end, color=green)

		# velocities
		vel_line_end = (car.location + car.velocity).as_list()
		r.draw_line_3d(pos, vel_line_end, color=red)

		vel_line_end = (ball.location + ball.velocity).as_list()
		r.draw_line_3d(ball.location.as_list(), vel_line_end, color=red)

		for p in state.get_all_players():
			if not p.player_id == self.index:
				vel_line_end = (p.location + p.velocity).as_list()
				r.draw_line_3d(p.location.as_list(), vel_line_end, color=red)

		# ball box
		r2 = Renderer(r)
		box_anchor = ball.location - Vector3(gi.BALL_SIZE/2, gi.BALL_SIZE/2, gi.BALL_SIZE/2)
		r2.draw_cube(box_anchor.as_list(), size=gi.BALL_SIZE, color=r2.red)

		# line to ball
		# r.draw_line_3d(pos, ball.location.as_list(), color=red)

		r.end_rendering()

	def retire(self):
		print("retire")
		if SAVE_NET:
			self.net.save()
		self.net.close()
		if SAVE_DATA:
			self.save(info_files=True)

	def save(self, info_files=False):
		run_indexer = ConfigParser()
		run_indexer.read(LOG_DIR + "run_index.cfg")

		try:
			run_indexer[self.name]["end_conditions"] = str(END_CONDITIONS).replace("[", "").replace("]", "")
			run_indexer[self.name]["bot_type"] = BOT_TYPE
			run_indexer[self.name]["reward"] = str(REWARDS).replace("[", "").replace("]", "")
			run_indexer[self.name]["reward_exp"] = str(self.reward_exp)
			run_indexer[self.name]["n_episodes"] = str(self.run_info.episode_count)
			run_indexer[self.name]["epsilon"] = str(self.epsilon)
			run_indexer[self.name]["epsilon_decay"] = str(self.epsilon_decay)
			run_indexer[self.name]["sarsa"] = str(self.sarsa)
			run_indexer[self.name]["relative_coordinates"] = str(self.rel_coords)
			run_indexer[self.name]["description"] = "- auto generated description -"
		except KeyError:
			run_indexer[self.name] = {
				"end_conditions": str(END_CONDITIONS).replace("[", "").replace("]", ""),
				"bot_type": BOT_TYPE,
				"reward": str(REWARDS).replace("[", "").replace("]", ""),
				"reward_exp": str(self.reward_exp),
				"n_episodes": str(self.run_info.episode_count),
				"epsilon": str(self.epsilon),
				"epsilon_decay": str(self.epsilon_decay),
				"sarsa": str(self.sarsa),
				"relative_coordinates": str(self.rel_coords),
				"description": "- auto generated description -"
			}

		with open(LOG_DIR + "run_index.cfg", "w") as ri_file:
			run_indexer.write(ri_file)

		if not os.path.isdir(LOG_DIR + self.name):
			os.makedirs(LOG_DIR + self.name)

		with open(LOG_DIR + self.name + "/state_composition.cfg", "w") as file:
			self.state_comp.write(file)

		if info_files:
			# copy the temp-files into logs folder
			for _, _, files in os.walk(TEMP_DIR):
				for file in files:
					with open(TEMP_DIR + file, "r") as src, open(LOG_DIR + self.name + "/" + file, "w") as dest:
						dest.write(src.read())

	def load(self, bot_name=LOAD_BOT_NAME, preserve=True):
		# read bot information from file
		run_indexer = ConfigParser()
		run_indexer.read(LOG_DIR + "run_index.cfg")
		net_info = run_indexer[bot_name]

		# determine the name for the bot
		if preserve:
			new_name = name_increment(bot_name)
			print("finding name")
			while os.path.isdir(LOG_DIR + new_name):
				print(new_name, "was not available")
				new_name = name_increment(new_name)
			print("name found")
		else:
			new_name = bot_name

		# reset attributes which may be set incorrectly in constructor
		self.name = new_name
		self.action_states = gi.get_action_states(net_info["bot_type"])		
		self.episode_end_condition = form_end_conditions(net_info["end_conditions"])
		self.epsilon = max(0.1, float(net_info["epsilon"]) - EPSILON_STARTUP_DECAY)
		self.epsilon_decay = float(net_info["epsilon_decay"])
		self.replay_memory = ReplayMemory(n_actions=len(self.action_states))
		self.state_comp.read(LOG_DIR + bot_name + "/state_composition.cfg")
		self.sarsa = net_info["sarsa"] == "True"
		self.reward_exp = int(net_info["reward_exp"])
		self.rel_coords = net_info["relative_coordinates"] == "True"

		self.net = NeuralNetwork.restore(bot_name, new_name=new_name, verbose=True)
		self.run_info.restore(bot_name)

		# copy information files from log into active (temp) folder
		bot_dir = LOG_DIR + bot_name + "/"
		for _, _, files in os.walk(bot_dir):
			for file in files:
				with open(bot_dir + file, "r") as src, open(TEMP_DIR + file, "w") as dest:
					dest.write(src.read())

	def __str__(self):
		return "FBv2_" + self.name + "(" + str(self.index) + ") " + ("blue" if self.team == 0 else "orange")


def name_increment(net_name):
	"""
	produces a new net name by incrementing its sub id character
	e.g.:
		- FlowBot15313548 becomes FlowBot15313548/a
		- FlowBot15313548/a becomes FlowBot15313548/b
		- etc.
	:param net_name: the old name
	:return: the new name
	"""
	new_name = net_name.split("_")
	if len(new_name) == 1:
		new_name = new_name[0] + "_a"
	elif len(new_name) == 2:
		new_name = new_name[0] + "_" + chr(ord(new_name[1]) + 1)
	else:
		raise ValueError("invalid net name")
	return new_name


def form_end_conditions(condition_string):
	conditions = condition_string.split(", ")
	try:
		conditions[0] = int(conditions[0])
	except (TypeError, ValueError):
		conditions[0] = None
	for i in range(1, len(conditions)):
		conditions[i] = conditions[i] == "True"

	return EpisodeEndCondition(fixed_length=conditions[0], goal=conditions[1], game_end=conditions[2], landed=conditions[3])


def clear_temp():
	for _, _, files in os.walk(TEMP_DIR):
		for file in files:
			with open(TEMP_DIR + file, "w") as tmp:
				tmp.write("")


class EpisodeEndCondition:
	def __init__(self, fixed_length=EC_FIXED_LENGTH, goal=EC_GOAL, game_end=EC_GAME_END, landed=EC_LANDED):
		if fixed_length is None and not goal and not game_end and not landed:
			raise ValueError("Unfulfillable end condition")

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
		self.was_met = False

	def is_met(self, state, player_id=0, remember=True):
		if remember:
			self.iteration_count += 1
		is_on_ground = state.get_player(player_id).is_on_ground

		fl_reached = self.fl is not None and self.iteration_count >= self.fl
		goal_scored = self.gs and not state.is_round_active and self.was_round_active
		game_ended = self.ge and state.is_match_ended
		has_landed = self.la and not self.was_on_ground and is_on_ground

		is_met = fl_reached or goal_scored or game_ended or has_landed
		if remember:
			self.was_round_active = state.is_round_active
			self.was_on_ground = is_on_ground

			self.was_met_fl = fl_reached
			self.was_met_gs = goal_scored
			self.was_met_ge = game_ended
			self.was_met_la = has_landed

			self.was_met = is_met
			if is_met:
				self.iteration_count = 0

		return is_met


class Renderer:
	def __init__(self, rlbot_renderer):
		self.rlbot_renderer = rlbot_renderer

		# todo add colors
		self.red = rlbot_renderer.create_color(255, 255, 0, 0)
		self.green = rlbot_renderer.create_color(255, 0, 255, 0)
		self.blue = rlbot_renderer.create_color(255, 0, 0, 255)
		self.white = rlbot_renderer.white()
		self.black = rlbot_renderer.black()

	def draw_cube(self, pos, size, color=None):
		if color is None:
			color = self.black

		self.draw_box(pos, size, size, size, color)

	def draw_box(self, pos, width, length, height, color=None):
		if color is None:
			color = self.black

		r = self.rlbot_renderer

		line_end = pos[:]
		line_end[0] += length
		r.draw_line_3d(pos, line_end, color)

		line_end = pos[:]
		line_end[1] += width
		r.draw_line_3d(pos, line_end, color)

		line_end = pos[:]
		line_end[2] += height
		r.draw_line_3d(pos, line_end, color)

		# ------

		line_start = pos[:]
		line_start[2] += height

		line_end = line_start[:]
		line_end[0] += length
		r.draw_line_3d(line_start, line_end, color)

		line_end = line_start[:]
		line_end[1] += width
		r.draw_line_3d(line_start, line_end, color)

		# ------

		line_start = pos[:]
		line_start[0] += length
		line_start[1] += width

		line_end = line_start[:]
		line_end[0] -= length
		r.draw_line_3d(line_start, line_end, color)

		line_end = line_start[:]
		line_end[1] -= width
		r.draw_line_3d(line_start, line_end, color)

		# ------

		line_start = pos[:]
		line_start[0] += length

		line_end = pos[:]
		line_end[0] += length
		line_end[2] += height
		r.draw_line_3d(line_start, line_end, color)

		# ------

		line_start = pos[:]
		line_start[1] += width

		line_end = pos[:]
		line_end[1] += width
		line_end[2] += height
		r.draw_line_3d(line_start, line_end, color)

		# ------

		line_start = pos[:]
		line_start[0] += length
		line_start[1] += width
		line_start[2] += height

		line_end = line_start[:]
		line_end[0] -= length
		r.draw_line_3d(line_start, line_end, color)

		line_end = line_start[:]
		line_end[1] -= width
		r.draw_line_3d(line_start, line_end, color)

		line_end = line_start[:]
		line_end[2] -= height
		r.draw_line_3d(line_start, line_end, color)


def log(*args):
	with open(LOG_DIR + "log.txt", "a") as log_file:
		s = ""
		for a in args:
			s += str(a) + " "

		s = s.rstrip(" ") + "\n"
		log_file.write(s)
