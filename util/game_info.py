import util.vector_math as vmath
import random

location_TOLERANCE = 500		# measurement inaccuracy in uu
BIG_BOOST_locationS = [
	[3075, -4100, 0],			# blue back left
	[-3075, -4100, 0],			# blue back right
	[-3600, 0, 0],				# mid blue right
	[3600, 0, 0],				# mid blue left
	[-3075, 4100, 0],			# orange back left
	[3075, 4100, 0]				# orange back right
]
BLUE_TEAM = 0
ORANGE_TEAM = 1

ALL_ACTION_STATES = [
	[0, 0, 0, 0, 0, 0, 0, 0, 0],

	[1, 0, 0, 0, 1, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 1, 0, 0, 0],
	[1, 0, 0, 0, 1, 0, 0, 1, 0],
	[1, 0, 0, 0, 1, 0, 0, 0, 1],
	[1, 0, 0, 0, 0, 1, 0, 0, 1],

	[0, 1, 0, 0, 1, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 1, 0, 0, 0],
	[0, 1, 0, 0, 1, 0, 0, 1, 0],
	[0, 1, 0, 0, 1, 0, 0, 0, 1],
	[0, 1, 0, 0, 0, 1, 0, 0, 1],

	[1, 0, 1, 0, 1, 0, 1, 0, 0],		# added acc
	[0, 1, 1, 0, 1, 0, 1, 0, 0],		# added acc
	[1, 0, 0, 0, 1, 0, 1, 0, 0],		# added acc
	[0, 1, 0, 0, 1, 0, 1, 0, 0],		# added acc
	[1, 0, 0, 1, 1, 0, 1, 0, 0],		# added acc
	[0, 1, 0, 1, 1, 0, 1, 0, 0],		# added acc
	[0, 0, 1, 0, 1, 0, 1, 0, 0],		# added acc
	[0, 0, 0, 1, 1, 0, 1, 0, 0],		# added acc
	[0, 0, 0, 0, 1, 0, 1, 0, 0],		# added acc

	[0, 0, 0, 0, 1, 0, 0, 0, 0],
	[0, 0, 0, 0, 1, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 1, 0, 0, 0],

	[0, 0, 0, 0, 0, 0, 0, 1, 0],
	[1, 0, 0, 0, 0, 0, 0, 1, 0],
	[0, 1, 0, 0, 0, 0, 0, 1, 0],
	[0, 0, 1, 0, 1, 0, 0, 1, 0],		# added acc
	[0, 0, 0, 1, 1, 0, 0, 1, 0],		# added acc
	[1, 0, 0, 0, 0, 0, 0, 1, 1],
	[0, 1, 0, 0, 0, 0, 0, 1, 1],

	[1, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 1, 0, 1, 0, 0, 0, 0],		# added acc
	[0, 0, 0, 1, 1, 0, 0, 0, 0],		# added acc
	[1, 0, 0, 0, 0, 0, 0, 0, 1],
	[0, 1, 0, 0, 0, 0, 0, 0, 1]
]

BOT_ACTION_STATES = {
	"grounded_simple": 	[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	"grounded": 		[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	"flying":			[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	"no_flip":			[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	"1%":				[0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
	"no_noop":			[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	"all":				[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}


ACTION_PERCENTAGES = {
	"000010000": 0.4052919438450752,
	"100010000": 0.15219938398407304,
	"010010000": 0.1513313451312783,
	"000010010": 0.07727151094199156,
	"000001000": 0.03699755088024476,
	"100010010": 0.020189454918659942,
	"010010010": 0.019724704930442552,
	"010001000": 0.012819251695487904,
	"100001000": 0.011153889359934934,
	"001010000": 0.00996535325236036,
	"001010100": 0.009992043949377822,

	"100000000": 0.00815709006007937,			# < 1%
	"010010001": 0.007655102371711216,
	"010000000": 0.007175473857025882,
	"100010001": 0.006424461019565235,
	"011010000": 0.004494934880571861,
	"000010100": 0.003633403166107866,
	"000110100": 0.00324254920245935,
	"000110000": 0.002829127174557734,
	"101010000": 0.0026410389520138237,
	"000010110": 0.0021468748067023226,
	"000000100": 0.0021342073672141176,
	"100110000": 0.0020253123187948204,
	"001010110": 0.001912172456334205,
	"011010110": 0.001771470862591381,
	"001010010": 0.0017656337506181288,
	"010010110": 0.0017388918162909751,
	"000100000": 0.001731592470325774,
	"101010110": 0.0016080199033457815,
	"000110110": 0.0015315856612814493,
	"000010001": 0.0015265998769153698,
	"000000010": 0.001513640778895078,
	"100110100": 0.0012101109562859748,
	"100000001": 0.0010955443318170428,
	"001000000": 0.0010777058653815676,
	"100010110": 0.001048957793313438,
	"010010100": 0.0009981303821000299,

	"000110010": 0.0008281446952030788,			# < 0.1%
	"011010010": 0.0008181770678024344,
	"100110110": 0.0008106727725984697,
	"100010100": 0.0007216617416707703,
	"101010010": 0.0006030946657144305,
	"100001001": 0.0006013801865055415,
	"000100100": 0.0005575604627252474,
	"101010100": 0.0004758093644476022,
	"011010100": 0.0004727114778770579,
	"011000000": 0.0004640484312077751,
	"100110001": 0.0004640444898762604,
	"010110001": 0.00046274779180792826,
	"010010011": 0.0004586133350490212,
	"101010001": 0.0004350047592760439,
	"000101100": 0.0004283833223313692,
	"010001001": 0.00042595546211832174,
	"100000010": 0.00042473759068028334,
	"010110000": 0.00041466748866025713,
	"001000010": 0.00040136155346667265,
	"010110100": 0.0004007782364024989,
	"010110110": 0.000349178324212212,
	"100100000": 0.0003395930059684923,
	"100110010": 0.00033645964741431584,
	"100010011": 0.0002947918906410411,
	"101010011": 0.00029193048396137806,
	"011010111": 0.0002883754029351301,
	"101000001": 0.00028722059280132667,
	"101010101": 0.0002846035486755743,
	"100010111": 0.0002821244511528359,
	"100100001": 0.0002805676252045344,
	"000010011": 0.0002776825705357833,
	"010000001": 0.0002584804033962264,
	"100100010": 0.000257767022392068,
	"011010001": 0.00023122215464064868,
	"100110111": 0.00022886129706335093,
	"000100010": 0.00022807303076041347,
	"000000001": 0.00022199943889628025,
	"101000000": 0.0002173959636871254,
	"010110010": 0.00021344280817789399,
	"100000011": 0.00019428793701651342,
	"011010011": 0.0001734658826244201,
	"100110101": 0.0001717474620840164,
	"001010001": 0.0001676011813305653,
	"010010101": 0.00016708092557062658,
	"101000010": 0.0001573418953978341,
	"101000110": 0.00015705023686574723,
	"100000100": 0.00014600268463007857,
	"100110011": 0.0001351088443234827,
	"001000111": 0.00012786073566797265,
	"010000011": 0.00012561811803611555,
	"001001000": 0.00011909915571082265,
	"010000110": 0.00011673829813352491,
	"000101000": 0.00011577267191242651,
	"000010101": 0.00011563078397789777,
	"001010101": 0.00011066864760090638,
	"010000010": 0.00011019962915065859,
	"010000100": 0.00010906452567442863,
	"001010011": 0.00010804766214363929,
	"100100100": 0.00010160752644864014,

	"010110101": 9.580588645902035e-05,				# < 0.001%
	"000001100": 9.422541252163072e-05,
	"001000100": 9.391404733197041e-05,
	"000000011": 9.363421279442762e-05,
	"100010101": 8.795475408176313e-05,
	"011000110": 8.132543447405898e-05,
	"000110001": 7.897245955979063e-05,
	"101010111": 7.656824733583134e-05,
	"011000011": 6.50792659705177e-05,
	"000001001": 6.466542616147551e-05,
	"101000011": 6.0751683967390966e-05,
	"000000110": 5.939192459482383e-05,
	"101000111": 5.629009669276488e-05,
	"001000011": 4.580221353218181e-05,
	"100000101": 4.0706071883691056e-05,
	"010010111": 3.862110751242144e-05,
	"010100000": 3.6934217624135254e-05,
	"010100110": 3.350920053787193e-05,
	"001000110": 3.042313796187173e-05,
	"101000100": 2.4861919194647868e-05,
	"010110011": 2.304890669789168e-05,
	"100001100": 1.8981452574734325e-05,
	"100101000": 1.4389801360123552e-05,
	"011000100": 1.2497962233073618e-05,
	"011000001": 1.1934351826473326e-05,
	"000110101": 1.168998927256271e-05,

	"010100010": 9.293659711632794e-06,				# < 0.0001%
	"000100110": 8.272854849328768e-06,
	"011001000": 8.229500202667208e-06,
	"100000110": 6.412546374396335e-06,
	"001000001": 5.529688115106366e-06,
	"000110011": 5.454802816327306e-06,
	"001010111": 3.7836782540998656e-06,
	"100101100": 3.492019722013001e-06,
	"010100100": 2.167732333078048e-06
}

MAX_VAL = 32767
MIN_VAL = -32768

MAX_VELOCITY = 2300
GRAVITATION = 650
CEILING_HEIGHT = 2000

N_BIG_BOOSTS = 6
N_SMALL_BOOSTS = 28


class GameInfo:
	def __init__(self, game_tick_packet):
		self.game_tick_packet = game_tick_packet

		self.time_played = game_tick_packet.game_info.seconds_elapsed
		self.time_remaining = game_tick_packet.game_info.game_time_remaining
		self.is_overtime = game_tick_packet.game_info.is_overtime
		self.is_round_active = game_tick_packet.game_info.is_round_active
		self.is_match_ended = game_tick_packet.game_info.is_match_ended

		self.ball_info = BallInfo(game_tick_packet.game_ball)

		self.num_players = game_tick_packet.num_cars
		self.orange_players = []
		self.blue_players = []
		# todo assignment of id may not be accurate
		player_id = 0
		for p_info_struct in game_tick_packet.game_cars:
			if p_info_struct.team == 0:
				self.blue_players.append(PlayerInfo(p_info_struct, player_id))
			else:
				self.orange_players.append(PlayerInfo(p_info_struct, player_id))
			player_id += 1

		self.boosts = []
		for boost in game_tick_packet.game_boosts:
			self.boosts.append(BoostInfo(boost))

	def dist_to_ball(self, player_id=0):
		player_pos = self.blue_players[player_id].location
		ball_pos = self.ball_info.location

		dist = vmath.dist(player_pos, ball_pos)
		flat_dist = vmath.dist(vmath.Vector3(player_pos.x, player_pos.y, 0), vmath.Vector3(ball_pos.x, ball_pos.y, 0))
		height_dist = abs(player_pos.z-ball_pos.z)

		return dist, flat_dist, height_dist

	def angle_to_ball(self, player_id=0):
		player_pos = self.blue_players[player_id].location
		ball_pos = self.ball_info.location

		facing = self.blue_players[player_id].get_facing()
		player_ball_vec = vmath.vec_between_points(ball_pos, player_pos)

		angle = vmath.angle(facing, player_ball_vec)
		horizontal_component = vmath.angle(vmath.Vector3(facing.x, facing.y, 0), vmath.Vector3(player_ball_vec.x, player_ball_vec.y, 0))
		vertical_component = vmath.angle(player_ball_vec, vmath.Vector3(player_ball_vec.x, player_ball_vec.y, facing.z))

		normal_vec = vmath.Vector3(facing.y, -facing.x, 0).normalize()
		if vmath.dist(player_pos, ball_pos) < vmath.dist(player_pos+normal_vec, ball_pos):
			sign_hc = -1
		else:
			sign_hc = 1

		return angle, sign_hc*horizontal_component, vertical_component

	def mirror(self):
		"""
			facing towards blue goal -> facing to neg. y
			facing towards blue's right -> facing to neg x

			x, y coordinates + teams need to be switched
		"""

		inv_dims = [True, True, False]
		self.ball_info.velocity.invert(inv_dims)
		self.ball_info.angular_velocity.invert(inv_dims)
		self.ball_info.location.invert(inv_dims)
		self.ball_info.rotation.invert(inv_dims)

		for player in self.blue_players:
			player.velocity.invert(inv_dims)
			player.angular_velocity.invert(inv_dims)
			player.location.invert(inv_dims)
			player.rotation.invert(inv_dims)
			player.team = ORANGE_TEAM

		for player in self.orange_players:
			player.velocity.invert(inv_dims)
			player.angular_velocity.invert(inv_dims)
			player.location.invert(inv_dims)
			player.rotation.invert(inv_dims)
			player.Team = BLUE_TEAM

		'''
		for boost in self.boost_pads + self.big_boosts:
			boost.location.invert(inv_dims)
		'''
		
		return self

	def get_all_players(self):
		return self.blue_players + self.orange_players

	def get_player(self, player_id):
		for player in self.get_all_players():
			if player.player_id == player_id:
				return player
		raise KeyError

	def get_state(self, player_id, state_composition):
		"""
		constructs a state (list of float values) that describes the current state of the game
		:param player_id: the id of the player requesting the state (this player has a specific
							position in the state)
		:param state_composition: defines which parts of the game state to include
		general:
			round, true_value, false_value, time_played, time_remaining, is_overtime, num_players
		ball:
			ball_location, ball_rotation, ball_velocity, ball_angular_velocity,
			ball_latest_touch, ball_lt_time, ball_lt_normal
		player:
			player_player_id, player_location, player_rotation, player_velocity,
			player_angular_velocity, player_score, player_is_demolished, player_is_on_ground,
			player_is_super_sonic, player_is_bot, player_jumped, player_double_jumped,
			player_team, player_boost
		teammates:
			mates_enable, mates_player_id, mates_location, mates_rotation, mates_velocity,
			mates_angular_velocity, mates_score, mates_is_demolished, mates_is_on_ground,
			mates_is_super_sonic, mates_is_bot, mates_jumped, mates_double_jumped,
			mates_team, mates_boost
		opponents:
			opps_enable, opps_player_id, opps_location, opps_rotation, opps_velocity,
			opps_angular_velocity, opps_score, opps_is_demolished, opps_is_on_ground,
			opps_is_super_sonic, opps_is_bot, opps_jumped, opps_double_jumped,
			opps_team, opps_boost
		boosts:
			bboost_enable, bboost_is_active, bboost_timer
		:return: representation of the current game state
		"""

		state = []

		state += self.ball_info.get_state(state_composition["ball"])
		state += self.get_player(player_id).get_state(state_composition["player"])

		if state_composition["mates"]["enable"] == "True":
			for p in self.blue_players:
				if p.player_id != player_id:
					state += p.get_state(state_composition["mates"])

		if state_composition["opponents"]["enable"] == "True":
			for p in self.orange_players:
				state += p.get_state(state_composition["opponents"])

		if state_composition["boosts"]["enable"] == "True":
			for b in self.boosts:
				state += b.get_state(give_is_active=state_composition["boosts"]["is_active"] == "True", give_timer=state_composition["boosts"]["timer"] == "True")

		t_val = float(state_composition["general"]["true_value"])
		f_val = float(state_composition["general"]["false_value"])
		n_dec = int(state_composition["general"]["round"])

		for i in range(len(state)):
			if isinstance(state[i], bool):
				state[i] = t_val if state[i] else f_val
			elif isinstance(state[i], float):
				state[i] = round(state[i], n_dec)
			elif isinstance(state[i], int):
				state[i] = float(state[i])
			else:
				raise ValueError("invalid value type in state")

		return state


class PlayerInfo:
	def __init__(self, p_info_struct, player_id):
		self.player_id = player_id
		self.location = vec3_struct_to_class(p_info_struct.physics.location)
		self.rotation = rot3_struct_to_class(p_info_struct.physics.rotation)
		self.velocity = vec3_struct_to_class(p_info_struct.physics.velocity)
		self.angular_velocity = vec3_struct_to_class(p_info_struct.physics.angular_velocity)
		self.score = ScoreInfo(p_info_struct.score_info)
		self.is_demolished = bool(p_info_struct.is_demolished)
		self.is_on_ground = bool(p_info_struct.has_wheel_contact)
		self.is_super_sonic = bool(p_info_struct.is_super_sonic)
		self.is_bot = bool(p_info_struct.is_bot)
		self.jumped = bool(p_info_struct.jumped)
		self.double_jumped = bool(p_info_struct.double_jumped)
		self.name = str(p_info_struct.name)
		self.team = int(p_info_struct.team)
		self.boost = int(p_info_struct.boost)

	def get_facing(self):
		x = 1 - 2 * (abs(self.rotation.y) / float(MAX_VAL))

		y = abs(self.rotation.y) / (MAX_VAL / 2)
		try:
			sign = self.rotation.y/abs(self.rotation.y)
		except ZeroDivisionError:
			sign = 0
		if abs(self.rotation.y) > MAX_VAL/2:
			y = 2-y
		y *= sign

		z = self.rotation.x/float(MAX_VAL/2)

		return vmath.Vector3(x, y, z)

	def get_state(self, options):
		"""
		:param options: id, location, rotation, velocity, angular_velocity,
			score, is_demolished, is_on_ground, is_super_sonic, is_bot, jumped,
			double_jumped, team, boost
		:return: float list representation of this player
		"""

		state = []

		if options["id"] == "True":
			state.append(self.player_id)
		if options["location"] == "True":
			state += self.location.as_list()
		if options["rotation"] == "True":
			state += self.rotation.as_list()
		if options["velocity"] == "True":
			state += self.velocity.as_list()
		if options["angular_velocity"] == "True":
			state += self.angular_velocity.as_list()
		if options["score"] == "True":
			state.append(self.score.score)
		if options["is_demolished"] == "True":
			state.append(self.is_demolished)
		if options["is_on_ground"] == "True":
			state.append(self.is_on_ground)
		if options["is_super_sonic"] == "True":
			state.append(self.is_super_sonic)
		if options["is_bot"] == "True":
			state.append(self.is_bot)
		if options["jumped"] == "True":
			state.append(self.jumped)
		if options["double_jumped"] == "True":
			state.append(self.double_jumped)
		if options["team"] == "True":
			state.append(self.team)
		if options["boost"] == "True":
			state.append(self.boost)

		return state


class BallInfo:
	def __init__(self, b_info_struct):
		self.location = vec3_struct_to_class(b_info_struct.physics.location)
		self.rotation = rot3_struct_to_class(b_info_struct.physics.rotation)
		self.velocity = vec3_struct_to_class(b_info_struct.physics.velocity)
		self.angular_velocity = vec3_struct_to_class(b_info_struct.physics.angular_velocity)
		self.latest_touch = vec3_struct_to_class(b_info_struct.latest_touch.hit_location)
		self.lt_player = str(b_info_struct.latest_touch.player_name)
		self.lt_time = float(b_info_struct.latest_touch.time_seconds)
		self.lt_normal = vec3_struct_to_class(b_info_struct.latest_touch.hit_normal)

	def get_state(self, options):
		"""
		:param options: location, rotation, velocity, angular_velocity, latest_touch,
			lt_time, lt_normal
		:return: list representation of the ball
		"""

		state = []

		if options["location"] == "True":
			state += self.location.as_list()
		if options["rotation"] == "True":
			state += self.rotation.as_list()
		if options["velocity"] == "True":
			state += self.velocity.as_list()
		if options["angular_velocity"] == "True":
			state += self.angular_velocity.as_list()
		if options["latest_touch"] == "True":
			state += self.latest_touch.as_list()
		if options["lt_time"] == "True":
			state.append(self.lt_time)
		if options["lt_normal"] == "True":
			state += self.lt_normal.as_list()

		return state


class BoostInfo:
	def __init__(self, boost_info_struct):
		# self.is_big = is_big
		# self.location = vec3_struct_to_class(boost_info_struct.location)
		self.is_active = bool(boost_info_struct.is_active)
		self.timer = int(boost_info_struct.timer)

	def get_state(self, give_is_active=True, give_timer=True):
		"""
		:param give_is_active: whether is_active will be included in the state
		:param give_timer: whether the timer will be included in the state
		:return: list representation of this boost pad
		"""
		state = []

		if give_is_active:
			state.append(self.is_active)
		if give_timer:
			state.append(self.timer)

		return state


class ScoreInfo:
	def __init__(self, score_info_struct):
		self.score = score_info_struct.score
		self.goals = score_info_struct.goals
		self.own_goals = score_info_struct.own_goals
		self.assists = score_info_struct.assists
		self.saves = score_info_struct.saves
		self.shots = score_info_struct.shots
		self.demolitions = score_info_struct.demolitions


def state_size(state_comp, n_opponents=0, n_mates=0):
	"""
	calculates how many values will be in a state given a state composition
	:param state_comp: dictionary specifying which information is contained in the state
	:param n_opponents: number of opponents in the game
	:param n_mates: number of teammates in the game
	:return: size of the state
	"""
	size = 0

	general_options = ["time_played", "time_remaining", "is_overtime", "num_players"]
	general_options = [state_comp["general"][opt] == "True" for opt in general_options]
	for i in general_options:
		if i:
			print("general option L+1")
			size += 1

	ball_vector_options = ["location", "rotation", "velocity", "angular_velocity", "latest_touch", "lt_normal"]
	ball_vector_options = [state_comp["ball"][opt] == "True" for opt in ball_vector_options]
	for i in ball_vector_options:
		if i:
			print("ball option L+3")
			size += 3

	if state_comp["ball"]["lt_time"] == "True":
		print("ball lt time L+1")
		size += 1

	size += player_state_size(state_comp["player"])
	if state_comp["mates"]["enable"] == "True":
		size += player_state_size(state_comp["mates"])*n_mates
	if state_comp["opponents"]["enable"] == "True":
		size += player_state_size(state_comp["opponents"])*n_opponents

	b_size = 0
	if state_comp["boosts"]["is_active"] == "True":
		b_size += 1
	if state_comp["boosts"]["timer"] == "True":
		b_size += 1

	if state_comp["boosts"]["enable"] == "True":
		size += b_size * N_BIG_BOOSTS

	return size


def player_state_size(p_state):
	size = 0

	if p_state["id"] == "True":
		size += 1
	if p_state["location"] == "True":
		size += 3
	if p_state["rotation"] == "True":
		size += 3
	if p_state["velocity"] == "True":
		size += 3
	if p_state["angular_velocity"] == "True":
		size += 3
	if p_state["score"] == "True":
		size += 1
	if p_state["is_demolished"] == "True":
		size += 1
	if p_state["is_on_ground"] == "True":
		size += 1
	if p_state["is_super_sonic"] == "True":
		size += 1
	if p_state["is_bot"] == "True":
		size += 1
	if p_state["jumped"] == "True":
		size += 1
	if p_state["double_jumped"] == "True":
		size += 1
	if p_state["team"] == "True":
		size += 1
	if p_state["boost"] == "True":
		size += 1

	return size


def vec3_struct_to_class(vec3):
	return vmath.Vector3(float(vec3.x), float(vec3.y), float(vec3.z))


def rot3_struct_to_class(rot3):
	return vmath.Vector3(float(rot3.pitch), float(rot3.yaw), float(rot3.roll))


def get_random_action(bot_type):
	actions = get_action_states(bot_type)
	probs = []
	for action_state in actions:
		probs.append(ACTION_PERCENTAGES[as_to_str(action_state)])

	for i in range(len(probs)-1):
		r = random.random()
		prob = sum(probs[i+1:]) / sum(probs[i:])
		if r > prob:
			return actions[i]
	return actions[-1]


def get_action_index(action, bot_type="all"):
	action_states = get_action_states(bot_type)
	for index in range(len(action_states)):
		match = True
		for act_index in range(9):
			if action_states[index][act_index] != action[act_index]:
				match = False
				break
		if match:
			return index

	raise ValueError("could not match action_state - ", action)


def get_action_states(bot_type):
	states = []

	selection = BOT_ACTION_STATES[bot_type]
	for index in range(len(ALL_ACTION_STATES)):
		if selection[index] != 0:
			states.append(ALL_ACTION_STATES[index])

	return states


def get_action_percentages(bot_type, normalize=False):
	actions = get_action_states(bot_type)
	res = {}
	for action in actions:
		as_str = as_to_str(action)
		try:
			res[as_str] = ACTION_PERCENTAGES[as_str]
		except KeyError:
			res[as_str] = 0.0

	if normalize:
		vector = vmath.Vector([res[key] for key in res])
		scaling = abs(vector)
		for key in res:
			res[key] = res[key]*scaling

	return res


def as_to_str(action_state):
	return str(action_state).replace(", ", "").replace("[", "").replace("]", "")


def print_action_states(bot_type):
	action_states = BOT_ACTION_STATES[bot_type]
	for index in range(len(action_states)):
		if action_states[index] == 1:
			action_state = ALL_ACTION_STATES[index]
			print(as_to_str(action_state))


if __name__ == '__main__':
	pass
