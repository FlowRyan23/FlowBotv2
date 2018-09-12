import util.game_info as gi
import numpy as np
from time import time
from Networks.legacy.XInputReader import get_xbox_output as get_controller_output
from util.data_processor_v3 import xbox_to_rlbot_controls
from util.game_info import GameInfo
from util.vector_math import Vector3, angle
from rlbot.agents.base_agent import BaseAgent
from Agents.FlowBotv2.flow_bot import Renderer
from configparser import ConfigParser

BOT_ROOT = str(__file__).replace("tdc_agent.py", "")
INFO_INTERVAL = 2


class Agent (BaseAgent):
	def __init__(self, name, team, index):
		super().__init__(name, team, index)
		self.name = name
		self.team = team
		self.index = index
		self.prev_time = time()
		self.state_composition = ConfigParser()
		self.state_composition.read(BOT_ROOT + "state_composition.cfg")
		self.sc_norm = {}
		for s in self.state_composition.keys():
			# print("section:", s)
			section = {}
			for key in self.state_composition[s]:
				# print("\tkey:", key)
				section[key] = self.state_composition[s][key]
			self.sc_norm[s] = section
		self.sc_norm["general"]["norm"] = "True"

	def get_output(self, game_tick_packet):
		controls = xbox_to_rlbot_controls(get_controller_output())
		# controls = nn_to_rlbot_controls(gi.get_random_action(self.bot_types[self.cur_bot], true_random=True))

		game_info = GameInfo(game_tick_packet)
		self.render(game_info)

		cur_time = time()
		if cur_time - self.prev_time >= INFO_INTERVAL:
			print(game_info.get_state(self.index, self.state_composition))
			print(game_info.get_state(self.index, self.sc_norm))
			self.prev_time = cur_time

		return controls

	def render(self, state):
		r = self.renderer

		car = state.get_player(self.index)
		ball = state.ball_info

		r.begin_rendering()

		# some default colors
		red = r.create_color(255, 255, 0, 0)
		green = r.create_color(255, 0, 255, 0)
		blue = r.create_color(255, 0, 0, 255)

		text_color = r.create_color(255, 255, 255, 255)

		own_car = state.get_player(self.index)
		info = {
			"Max Boost Timer": str(max([b.timer for b in state.boosts])),
		}
		for i, key in enumerate(info):
			r.draw_string_2d(32, i*16 + 16, 1, 1, key + ": " + info[key], color=text_color)

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
