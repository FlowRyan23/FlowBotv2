import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


def height():
	x = [i * 20 for i in range(101)]
	y1 = [(20 * i) / 2000.0 for i in range(101)]
	y2 = [((20 * i) / 2000.0) ** 2 for i in range(101)]
	y3 = [((20 * i) / 2000.0) ** 3 for i in range(101)]
	y4 = [((20 * i) / 2000.0) ** 4 for i in range(101)]

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


def angle_to():
	x = [i - 180 for i in range(361)]
	y1 = [max(0.0, (((180 - abs(x[i])) / 180.0) - 0.5) * 2) for i in range(361)]
	y2 = []
	for i in range(361):
		if y1[i] < 0:
			y2.append(-(y1[i] ** 2))
		else:
			y2.append(y1[i] ** 2)

	y3 = []
	for i in range(361):
		y3.append(y1[i] ** 3)

	y4 = []
	for i in range(361):
		if y1[i] < 0:
			y4.append(-(y1[i] ** 4))
		else:
			y4.append(y1[i] ** 4)

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


def bool_height():
	x = [i * 20 for i in range(101)]
	y = [0 for _ in range(101)]

	y[int(len(y)/2):] = [1 for _ in range(int(len(y)/2)+1)]

	plt.plot(x, y)
	plt.show()


def bool_angle():
	x = [i - 180 for i in range(361)]
	y = [0 for _ in range(361)]

	y[int(len(y) / 2) - 60: int(len(y) / 2) + 60] = [1 for _ in range(120)]

	plt.plot(x, y)
	plt.show()


def discrete_height(step_size=20):
	x = [i * 20 for i in range(101)]
	y1 = [(20 * step_size * int(i / step_size)) / 2000.0 for i in range(101)]
	y2 = [y1[i] ** 2 for i in range(101)]
	y3 = [y1[i] ** 3 for i in range(101)]
	y4 = [y1[i] ** 4 for i in range(101)]

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


def discrete_angle(step_size=20, no_neg=False):
	x = [i - 180 for i in range(361)]
	y1 = [(((180 - abs(x[i])) / 180.0) - 0.5) * 2 for i in range(361)]
	if no_neg:
		y1 = [max(0.0, y1[i]) for i in range(361)]

	n_sections = int(len(x) / step_size) + 1
	for a in range(n_sections - 1):
		y1[a * step_size: (a+1) * step_size] = [y1[a * step_size] for _ in range(step_size)]

	x = [x[i] - step_size/2 for i in range(361)]

	y2 = []
	for i in range(361):
		if y1[i] < 0:
			y2.append(-(y1[i] ** 2))
		else:
			y2.append(y1[i] ** 2)

	y3 = []
	for i in range(361):
		y3.append(y1[i] ** 3)

	y4 = []
	for i in range(361):
		if y1[i] < 0:
			y4.append(-(y1[i] ** 4))
		else:
			y4.append(y1[i] ** 4)

	plt.plot(x, y1, x, y2, x, y3, x, y4)
	plt.show()


if __name__ == '__main__':
	a = ["RE_HEIGHT", "RE_AIRTIME", "RE_BALL_DIST", "RE_SS_DIFF", "RE_FACING_UP", "RE_FACING_OPP", "RE_FACING_BALL"]
	for s in a:
		print("[\"" + s.lower() + "\"] += ")
