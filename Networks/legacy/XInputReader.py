# Thanks to https://github.com/r4dian/Xbox-360-Controller-for-Python/blob/master/xinput.py

import ctypes
from time import sleep

# structs according to
# http://msdn.microsoft.com/en-gb/library/windows/desktop/ee417001%28v=vs.85%29.aspx


class XINPUT_GAMEPAD(ctypes.Structure):
	_fields_ = [
		('buttons', ctypes.c_ushort),  # wButtons
		('left_trigger', ctypes.c_ubyte),  # bLeftTrigger
		('right_trigger', ctypes.c_ubyte),  # bLeftTrigger
		('l_thumb_x', ctypes.c_short),  # sThumbLX
		('l_thumb_y', ctypes.c_short),  # sThumbLY
		('r_thumb_x', ctypes.c_short),  # sThumbRx
		('r_thumb_y', ctypes.c_short),  # sThumbRy
	]


class XINPUT_STATE(ctypes.Structure):
	_fields_ = [
		('packet_number', ctypes.c_ulong),  # dwPacketNumber
		('gamepad', XINPUT_GAMEPAD),  # Gamepad
	]


class XINPUT_VIBRATION(ctypes.Structure):
	_fields_ = [("wLeftMotorSpeed", ctypes.c_ushort),
				("wRightMotorSpeed", ctypes.c_ushort)]


xinput = ctypes.windll.xinput9_1_0  # Check you C:\Windows\System32 folder for xinput dlls
# xinput1_2, xinput1_1 (32-bit Vista SP1)
# xinput1_3 (64-bit Vista SP1)

ERROR_DEVICE_NOT_CONNECTED = 1167
ERROR_SUCCESS = 0


def get_state(device_number):
	# Get the state of the controller represented by this object
	state = XINPUT_STATE()
	res = xinput.XInputGetState(device_number, ctypes.byref(state))
	if res == ERROR_SUCCESS:
		return state
	if res != ERROR_DEVICE_NOT_CONNECTED:
		raise RuntimeError(
			"Unknown error %d attempting to get state of device %d" % (res, device_number))
	# else return None (device is not connected)


def get_xbox_output():
	state = get_state(0)

	# output = [l_thumb_x, l_thumb_y, right_trigger, left_trigger, B, A, right_sbutton] (len=10)
	output = [state.gamepad.l_thumb_x,
			state.gamepad.l_thumb_y,
			# state.gamepad.r_thumb_x,
			# state.gamepad.r_thumb_y,
			state.gamepad.right_trigger,
			state.gamepad.left_trigger]

	buttons = state.gamepad.buttons
	if buttons >= 32768:
		# y is pressed
		# output.append(1.0)
		buttons = buttons - 32768
	# else:
		# output.append(-1.0)

	if buttons >= 16384:
		# x is pressed
		# output.append(1.0)
		buttons = buttons - 16384
	# else:
		# output.append(-1.0)

	if buttons >= 8192:
		# b is pressed
		output.append(1.0)
		buttons = buttons - 8192
	else:
		output.append(-1.0)

	if buttons >= 4096:
		# a is pressed
		output.append(1.0)
		buttons = buttons - 4096
	else:
		output.append(-1.0)

	if buttons >= 2048:
		# ? is pressed
		# output.append(1.0)
		buttons = buttons - 2048
	# else:
		# output.append(-1.0)

	if buttons >= 1024:
		# ? is pressed
		# output.append(1.0)
		buttons = buttons - 1024
	# else:
		# output.append(-1.0)

	if buttons >= 512:
		# right_sButton is pressed
		output.append(1.0)
		buttons = buttons - 512
	else:
		output.append(-1.0)

	if buttons >= 256:
		# left_sButton is pressed
		# output.append(1.0)
		buttons = buttons - 256
	# else:
		# output.append(-1.0)

	return output


def all_xbox_output():
	state = get_state(0)

	# output = [l_thumb_x, l_thumb_y, right_trigger, left_trigger, B, A, right_sbutton] (len=10)
	output = [state.gamepad.l_thumb_x,
			state.gamepad.l_thumb_y,
			state.gamepad.r_thumb_x,
			state.gamepad.r_thumb_y,
			state.gamepad.right_trigger,
			state.gamepad.left_trigger]

	buttons = state.gamepad.buttons
	for i in range(16):
		if buttons >= (1 << (15-i)):
			# button is pressed
			output.append(1.0)
			buttons -= (1 << (15-i))
		else:
			output.append(0.0)

	return output


if __name__ == "__main__":
	while True:
		sleep(5)
		print(all_xbox_output())
