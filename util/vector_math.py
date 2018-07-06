import math
import numpy as np


class Vector3:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def invert(self, dims=[True, True, True]):
		if dims[0]:
			self.x = -self.x
		if dims[1]:
			self.y = -self.y
		if dims[2]:
			self.z = -self.z

	def as_list(self):
		return [self.x, self.y, self.z]
	
	def __add__(self, other):
		return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
	
	def __sub__(self, other):
		return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

	def __mul__(self, other):
		return self.x*other.x + self.y*other.y + self.z*other.z
	
	def __abs__(self):
		return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2))

	def __eq__(self, other):
		# bad practice since coordinates are often float/double
		return self.x == other.x and self.y == other.y and self.z == other.z

	def __floor__(self):
		return Vector3(math.floor(self.x), math.floor(self.y), math.floor(self.z))

	def __ceil__(self):
		return Vector3(math.ceil(self.x), math.ceil(self.y), math.ceil(self.z))

	def __invert__(self):
		return Vector3(-self.x, -self.y, -self.z)

	def __str__(self):
		return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)

	def scalar_mul(self, val):
		return Vector3(self.x*val, self.y*val, self.z*val)

	def cross_mul(self, other):
		x = self.y*other.z - self.z*other.y
		y = self.z*other.x - self.x*other.z
		z = self.x*other.y - self.y*other.x
		return Vector3(x, y, z)

	def normalize(self):
		return self.scalar_mul(1/abs(self))


class Vector:
	def __init__(self, values):
		self.values = values
		self.size = len(values)

	def as_list(self):
		return self.values

	def __add__(self, other):
		if other.size != self.size:
			raise ArithmeticError("self " + str(self.size) + "other " + str(other.size))

		res_list = np.zeros([self.size])
		for i in range(self.size):
			res_list[i] = self.values[i] + other.values[i]

		return Vector(res_list)

	def __sub__(self, other):
		if other.size != self.size:
			raise ArithmeticError

		res_list = np.zeros([self.size])
		for i in range(self.size):
			res_list[i] = self.values[i] - other.values[i]

		return Vector(res_list)

	def __mul__(self, other):
		if other.size != self.size:
			raise ArithmeticError

		return sum([self.values[i]*other.values[i] for i in range(self.size)])

	def __abs__(self):
		return math.sqrt(sum(math.pow(self.values[i], 2) for i in range(self.size)))

	def __eq__(self, other):
		# bad practice since coordinates are often float/double
		for i in range(self.size):
			if self.values[i] != other.values[i]:
				return False
		return True

	def __floor__(self):
		return Vector([math.floor(self.values[i]) for i in range(self.size)])

	def __ceil__(self):
		return Vector([math.ceil(self.values[i]) for i in range(self.size)])

	def __invert__(self):
		for i in range(self.size):
			self.values[i] = -self.values[i]
		return self

	def __len__(self):
		return self.size

	def __str__(self):
		return "Vector(" + str(self.size) + "), " + str(self.values)

	def scalar_mul(self, val):
		res_list = np.zeros([self.size])
		for i in range(self.size):
			res_list[i] = self.values[i] * val
		return Vector(res_list)

	def normalize(self):
		return self.scalar_mul(1 / abs(self))

		
def dist(vec_a, vec_b):
	delta_x = vec_a.x - vec_b.x
	delta_y = vec_a.y - vec_b.y
	delta_z = vec_a.z - vec_b.z
	return math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2) + math.pow(delta_z, 2))


def angle(vec_a, vec_b, in_deg=True):
	try:
		radians = math.acos((vec_a*vec_b)/(abs(vec_a)*abs(vec_b)))
	except ZeroDivisionError:
		return 0
	except ValueError:
		val = min(1, max((vec_a*vec_b)/(abs(vec_a)*abs(vec_b)), -1))
		radians = math.acos(val)
		print("corrected value error for angle between", vec_a, "and", vec_b)
	degrees = math.degrees(radians)
	if in_deg:
		return degrees
	else:
		return radians


def vec_between_points(point_a, point_b):
	x = point_a.x - point_b.x
	y = point_a.y - point_b.y
	z = point_a.z - point_b.z
	# returned Vector is the one from b to a
	return Vector3(x, y, z)