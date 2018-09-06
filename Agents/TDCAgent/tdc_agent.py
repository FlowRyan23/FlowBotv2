import util.game_info as gi
import numpy as np
from time import time
from Networks.legacy.XInputReader import get_xbox_output as get_controller_output
from util.data_processor_v3 import xbox_to_rlbot_controls, nn_to_rlbot_controls
from util.data_processor_v3 import array_to_scs
from util.game_info import GameInfo
from util.vector_math import Vector3, angle
from rlbot.agents.base_agent import BaseAgent
from Agents.FlowBotv2.flow_bot import Renderer, EpisodeEndCondition

INFO_INTERVAL = 5.0  # in seconds
RE_HEIGHT = True						# car.z / ceiling_height
RE_AIRTIME = True						# +1 for every iteration where !car.on_ground (given when landed)
RE_BALL_DIST = True						# distance between car and ball
RE_FACING_UP = True						# angle between z-axes and car.facing (normalized to 0-1)
RE_FACING_OPP = True					# angle between y-axes and car.facing (normalized to 0-1)
RE_FACING_BALL = True					# angle between car->ball and car.facing (normalized to 0-1)
REWARDS = [RE_HEIGHT, RE_AIRTIME, RE_BALL_DIST, RE_FACING_UP, RE_FACING_OPP, RE_FACING_BALL]
REWARD_EXP = 1
ALLOW_NEGATIVE_REWARD = True


class Agent (BaseAgent):
	def __init__(self, name, team, index):
		super().__init__(name, team, index)
		self.name = name
		self.team = team
		self.index = index
		self.prev_time = time()

		self.reward_accumulator = 0
		self.episode_end_condition = EpisodeEndCondition(fixed_length=5000, goal=False, game_end=False, landed=False)
		self.iteration_count = 0
		self.bot_types = ["grounded_simple", "grounded", "flying", "no_flip", "no_jmp_boost", "1p", "no_noop", "all"]
		self.cur_bot = 0
		self.reward_data = {"re_height": [], "re_airtime": [], "re_ball_dist": [], "re_facing_up": [], "re_facing_opp": [], "re_facing_ball": []}

	def get_output(self, game_tick_packet):
		self.iteration_count += 1

		if self.iteration_count % 5000 == 0:
			print(self.bot_types[self.cur_bot])
			for key in self.reward_data:
				avrg_reward = sum(self.reward_data[key]) / len(self.reward_data[key])
				print(key + ":", avrg_reward)
				print()
			self.reward_data = {"re_height": [], "re_airtime": [], "re_ball_dist": [], "re_facing_up": [], "re_facing_opp": [], "re_facing_ball": []}

			self.cur_bot += 1
			if self.cur_bot >= len(self.bot_types):
				print("done")
				return nn_to_rlbot_controls(gi.get_action_states("all")[0])

		# controls = xbox_to_rlbot_controls(get_controller_output())
		controls = nn_to_rlbot_controls(gi.get_random_action(self.bot_types[self.cur_bot], true_random=True))

		game_info = GameInfo(game_tick_packet)
		reward, h, at, bd, fu, fo, fb = self.reward(game_info)
		self.reward_data["re_height"].append(h)
		self.reward_data["re_airtime"].append(at)
		self.reward_data["re_ball_dist"].append(bd)
		self.reward_data["re_facing_up"].append(fu)
		self.reward_data["re_facing_opp"].append(fo)
		self.reward_data["re_facing_ball"].append(fb)
		self.render(game_info, reward, h, at, bd, fu, fo, fb)
		self.episode_end_condition.is_met(game_info)

		new_time = time()
		if new_time - self.prev_time > INFO_INTERVAL:
			# print("")
			# print(self,)
			# print(controls)
			# print("Rotation:", game_info.get_player(self.index).rotation)
			# base_x, base_y, base_z = game_info.get_player(self.index).get_basis()
			# base_x = round_list(base_x, 2)
			# base_y = round_list(base_y, 2)
			# base_z = round_list(base_z, 2)
			# print("Basis x:", base_x)
			# print("Basis y:", base_y)
			# print("Basis z:", base_z)
			# fb_component = (180 - game_info.angle_to_ball(self.index)[0]) / 180
			# fb_component = (abs(fb_component) - 0.5) * 2
			# print(fb_component)
			# print("Game info array:", game_info_array)
			# print("Game info mirror", game_info_as_array(game_info.mirror())[0])
			# print("Location:", game_info.get_player(self.index).location)
			# print("Velocity:", abs(game_info.blue_players[0].Velocity))
			# print("Ball Location:", game_info.ball_info.Location)
			# facing = game_info.get_player(self.index).get_facing()
			# print("Facing:", facing)
			# car = game_info.get_player(self.index)
			# print("is on ground" if car.is_on_ground else "is in the air")
			# print("norm facing:", facing.normalize())
			# print("angle to up:", vmath.angle(facing, vmath.Vector3(0, 0, 1)))
			# print("angle to up:", vmath.angle(vmath.Vector3(base_x[0], base_x[1], base_x[2]), vmath.Vector3(0, 0, 1)))
			# print("Angle to ball:", game_info.angle_to_ball(0)[1])
			# print("Dist from ball:", game_info.dist_to_ball(0))
			# print("Fitness:", self.fitness(game_info))
			# print(gp_logstring, "\n")
			self.prev_time = new_time

		return controls

	def render(self, state, reward, h_component, at_component, bd_component, fu_component, fo_component, fb_component):
		r = self.renderer

		car = state.get_player(self.index)
		ball = state.ball_info

		r.begin_rendering()

		# some default colors
		red = r.create_color(255, 255, 0, 0)
		green = r.create_color(255, 0, 255, 0)
		blue = r.create_color(255, 0, 0, 255)

		text_color = r.create_color(255, 255, 255, 255)
		r.draw_string_2d(8, 16, 1, 1, "Reward: " + str("{0:.3f}").format(reward), text_color)
		r.draw_string_2d(8, 32, 1, 1, "h_component: " + str("{0:.3f}").format(h_component), text_color)
		r.draw_string_2d(8, 48, 1, 1, "at_component: " + str("{0:.3f}").format(at_component), text_color)
		r.draw_string_2d(8, 64, 1, 1, "bd_component: " + str("{0:.3f}").format(bd_component), text_color)
		r.draw_string_2d(8, 96, 1, 1, "fu_component: " + str("{0:.3f}").format(fu_component), text_color)
		r.draw_string_2d(8, 112, 1, 1, "fo_component: " + str("{0:.3f}").format(fo_component), text_color)
		r.draw_string_2d(8, 128, 1, 1, "fb_component: " + str("{0:.3f}").format(fb_component), text_color)

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

		# line to ball
		# r.draw_line_3d(pos, ball.location.as_list(), color=red)

		# ball box
		# r2 = Renderer(r)
		# box_anchor = ball.location - Vector3(gi.BALL_SIZE/2, gi.BALL_SIZE/2, gi.BALL_SIZE/2)
		# r2.draw_cube(box_anchor.as_list(), size=gi.BALL_SIZE, color=r2.red)

		r2 = Renderer(r)
		basis = np.transpose(car.get_basis())
		ball_pos = np.matmul(basis, ball.get_relative(basis, offset=car.location).location.as_list())
		ball_pos = Vector3.from_list(ball_pos) + car.location - Vector3(gi.BALL_SIZE/2, gi.BALL_SIZE/2, gi.BALL_SIZE/2)
		r2.draw_cube(ball_pos.as_list(), size=gi.BALL_SIZE, color=r2.green)

		r.end_rendering()

	def __str__(self):
		return "TDC(" + str(self.index) + ") " + ("blue" if self.team == 0 else "orange")

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
				if self.iteration_count % 5000 <= 0:
					at_component = 0
				else:
					at_component = self.reward_accumulator / (self.iteration_count % 5000)
				reward += at_component
				self.reward_accumulator = 0
			else:
				if not car.is_on_ground:
					self.reward_accumulator += 1

		if RE_HEIGHT:
			h_component = (car.location.z - gi.FLOOR_HEIGHT) / gi.ARENA_HEIGHT
			reward += h_component

		if RE_FACING_UP:
			vertical = Vector3(0, 0, -1)
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

		if not ALLOW_NEGATIVE_REWARD:
			reward = max(0, reward)
		n_enabled_rewards = len([b for b in REWARDS if b])
		reward /= n_enabled_rewards

		is_neg = reward < 0		# raising reward to an even power would result in loss of sign
		reward = abs(reward) ** REWARD_EXP
		if ALLOW_NEGATIVE_REWARD and is_neg:
			reward = -reward

		return reward, h_component, at_component, bd_component, fu_component, fo_component, fb_component


def round_list(l, n):
	for i in range(len(l)):
		l[i] = round(l[i], n)

	return l
