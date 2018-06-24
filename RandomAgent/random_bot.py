from rlbot.agents.base_agent import BaseAgent
from util.data_processor_v3 import nn_to_rlbot_controls
import util.game_info as gi


class RandomAgent(BaseAgent):
	"""
	This agent returns a random action using the game_info.get_random_action() method
	The likelihood of an action being chosen is based on how often a human chooses the action
	Is used as a teacher for the Neural Network Agent
	"""

	def __init__(self, name, team, index, bot_type):
		super().__init__(name, team, index)
		self.bot_type = bot_type

	def get_output_vector(self, game_tick_packet):
		return nn_to_rlbot_controls(gi.get_random_action(bot_type=self.bot_type))