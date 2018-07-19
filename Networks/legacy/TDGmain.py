import time
import ctypes
import mmap
import neural_net.XInputReader as gpad_input
import game_data_struct as gd


# todo rewrite for new game_tick_packet format
def get_game_info(game_tick_packet):
	# [BallX, -Y, -Z, BallRotX, -Y, -Z, BallVelX, -Y, -Z, BallAngVelX, -Y, -Z, BallAccX, -Y, -Z] entries 0-14 (SelfInfo)
	# [SelfX, -Y, -Z, SelfRotX, -Y, -Z, SelfVelX, -Y, -Z, SelfAngVelX, -Y, -Z, Boost] entries 15-27 (PlayerInfo self)
	# [OppX, -Y, -Z, OppRotX, -Y, -Z, OppVelX, -Y, -Z, OppAngVelX, -Y, -Z, Boost] entries 28-40 (PlayerInfo opponent)
	current_game_info = [
		round(game_tick_packet.gameball.Location.X, 2),
		round(game_tick_packet.gameball.Location.Y, 2),
		round(game_tick_packet.gameball.Location.Z, 2),
		round(game_tick_packet.gameball.Rotation.Pitch, 2),
		round(game_tick_packet.gameball.Rotation.Yaw, 2),
		round(game_tick_packet.gameball.Rotation.Roll, 2),
		round(game_tick_packet.gameball.Velocity.X, 2),
		round(game_tick_packet.gameball.Velocity.Y, 2),
		round(game_tick_packet.gameball.Velocity.Z, 2),
		round(game_tick_packet.gameball.AngularVelocity.X, 2),
		round(game_tick_packet.gameball.AngularVelocity.Y, 2),
		round(game_tick_packet.gameball.AngularVelocity.Z, 2),
		round(game_tick_packet.gameball.Acceleration.X, 2),
		round(game_tick_packet.gameball.Acceleration.Y, 2),
		round(game_tick_packet.gameball.Acceleration.Z, 2),

		round(game_tick_packet.gamecars[0].Location.X, 2),
		round(game_tick_packet.gamecars[0].Location.Y, 2),
		round(game_tick_packet.gamecars[0].Location.Z, 2),
		round(game_tick_packet.gamecars[0].Rotation.Pitch, 2),
		round(game_tick_packet.gamecars[0].Rotation.Yaw, 2),
		round(game_tick_packet.gamecars[0].Rotation.Roll, 2),
		round(game_tick_packet.gamecars[0].Velocity.X, 2),
		round(game_tick_packet.gamecars[0].Velocity.Y, 2),
		round(game_tick_packet.gamecars[0].Velocity.Z, 2),
		round(game_tick_packet.gamecars[0].AngularVelocity.X, 2),
		round(game_tick_packet.gamecars[0].AngularVelocity.Y, 2),
		round(game_tick_packet.gamecars[0].AngularVelocity.Z, 2),
		game_tick_packet.gamecars[0].Boost,

		round(game_tick_packet.gamecars[1].Location.X, 2),
		round(game_tick_packet.gamecars[1].Location.Y, 2),
		round(game_tick_packet.gamecars[1].Location.Z, 2),
		round(game_tick_packet.gamecars[1].Rotation.Pitch, 2),
		round(game_tick_packet.gamecars[1].Rotation.Yaw, 2),
		round(game_tick_packet.gamecars[1].Rotation.Roll, 2),
		round(game_tick_packet.gamecars[1].Velocity.X, 2),
		round(game_tick_packet.gamecars[1].Velocity.Y, 2),
		round(game_tick_packet.gamecars[1].Velocity.Z, 2),
		round(game_tick_packet.gamecars[1].AngularVelocity.X, 2),
		round(game_tick_packet.gamecars[1].AngularVelocity.Y, 2),
		round(game_tick_packet.gamecars[1].AngularVelocity.Z, 2),
		game_tick_packet.gamecars[1].Boost
	]

	return current_game_info


RAW_DATA_PATH = '../../FlowBot Training Data/raw/training_data_'
SAMPLE_RATE = 100  # per second (approximated)

SHARED_MEMORY_TAG = 'Local\\RLBotOutput'
REFRESH_IN_PROGRESS = 1


if __name__ == '__main__':
	# Set up shared memory for game data
	game_data_shared_memory = mmap.mmap(-1, ctypes.sizeof(gd.GameTickPacketWithLock), SHARED_MEMORY_TAG)
	# This lock ensures that a read cannot start while the dll is writing to shared memory.
	lock = ctypes.c_long(0)
	game_tick_packet = gd.GameTickPacket()

	file_path = RAW_DATA_PATH + str(int(time.time())) + '.txt'

	count = 0
	discarded = 0
	with open(file_path, 'a') as tdFile:
		while True:
			# Read from game data shared memory
			game_data_shared_memory.seek(0)  # Move to beginning of shared memory
			ctypes.memmove(ctypes.addressof(lock), game_data_shared_memory.read(ctypes.sizeof(lock)), ctypes.sizeof(lock))  # dll uses InterlockedExchange so this read will return the correct value!

			# copy shared memory into struct
			if lock.value != REFRESH_IN_PROGRESS:
				ctypes.memmove(ctypes.addressof(game_tick_packet), game_data_shared_memory.read(ctypes.sizeof(gd.GameTickPacket)), ctypes.sizeof(gd.GameTickPacket))

			game_info = get_game_info(game_tick_packet)
			gamepad_output = gpad_input.get_xbox_output()
			tdFile.write(str(game_info))
			tdFile.write('\n')
			tdFile.write(str(gamepad_output))
			tdFile.write('\n\n')
			count += 1
			# print_gameInfo(blueInputs.GameTickPacket)
			# print(XInputReader.get_xbox_output())
			# print()

			if count % 1000 == 0:
				print('+', count / 1000)
			time.sleep(1 / SAMPLE_RATE)
