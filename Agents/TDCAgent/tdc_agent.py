import util.game_info as gi
import numpy as np
from time import time
from Networks.legacy.XInputReader import get_xbox_output as get_controller_output
from util.data_processor_v3 import xbox_to_rlbot_controls
from util.data_processor_v3 import array_to_scs
from util.game_info import GameInfo
from util.vector_math import Vector3
from rlbot.agents.base_agent import BaseAgent
from Agents.FlowBotv2.flow_bot import Renderer

INFO_INTERVAL = 5.0  # in seconds


class Agent (BaseAgent):
	def __init__(self, name, team, index):
		super().__init__(name, team, index)
		self.name = name
		self.team = team
		self.index = index
		self.prev_time = time()

	def get_output(self, game_tick_packet):
		controls = xbox_to_rlbot_controls(get_controller_output())
		game_info = GameInfo(game_tick_packet)
		# self.render(game_info)

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
			car = game_info.get_player(self.index)
			print("is on ground" if car.is_on_ground else "is in the air")
			# print("norm facing:", facing.normalize())
			# print("angle to up:", vmath.angle(facing, vmath.Vector3(0, 0, 1)))
			# print("angle to up:", vmath.angle(vmath.Vector3(base_x[0], base_x[1], base_x[2]), vmath.Vector3(0, 0, 1)))
			# print("Angle to ball:", game_info.angle_to_ball(0)[1])
			# print("Dist from ball:", game_info.dist_to_ball(0))
			# print("Fitness:", self.fitness(game_info))
			# print(gp_logstring, "\n")
			self.prev_time = new_time

		return array_to_scs(controls)

	def render(self, state):
		r = self.renderer

		car = state.get_player(self.index)
		ball = state.ball_info

		r.begin_rendering()

		# some default colors
		red = r.create_color(255, 255, 0, 0)
		green = r.create_color(255, 0, 255, 0)
		blue = r.create_color(255, 0, 0, 255)

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


def round_list(l, n):
	for i in range(len(l)):
		l[i] = round(l[i], n)

	return l
