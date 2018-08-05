from time import time
from Networks.legacy.XInputReader import get_xbox_output as get_controller_output
from util.data_processor_v3 import xbox_to_rlbot_controls
from util.data_processor_v3 import array_to_scs
from util.game_info import GameInfo
from rlbot.agents.base_agent import BaseAgent

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

		new_time = time()
		if new_time - self.prev_time > INFO_INTERVAL:
			# print(self,)
			# print(controls)
			print("Rotation:", game_info.get_player(self.index).rotation)
			base_x, base_y, base_z = game_info.get_player(self.index).get_basis()
			print("Basis x:", base_x, 2)
			print("Basis y:", base_y, 2)
			print("Basis z:", base_z, 2)
			# print("Game info array:", game_info_array)
			# print("Game info mirror", game_info_as_array(game_info.mirror())[0])
			# print("Location:", game_info.blue_players[0].Location)
			# print("Velocity:", abs(game_info.blue_players[0].Velocity))
			# print("Ball Location:", game_info.ball_info.Location)
			# print("Facing:", game_info.blue_players[0].get_facing())
			# print("Angle to ball:", game_info.angle_to_ball(0)[1])
			# print("Dist from ball:", game_info.dist_to_ball(0))
			# print("Fitness:", self.fitness(game_info))
			# print(gp_logstring, "\n")
			self.prev_time = new_time

		return array_to_scs(controls)

	def __str__(self):
		return "TDC(" + str(self.index) + ") " + ("blue" if self.team == 0 else "orange")
