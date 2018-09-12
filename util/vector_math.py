import math
import numpy as np
from scipy.linalg import expm


class Vector3:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	@staticmethod
	def from_list(l):
		return Vector3(l[0], l[1], l[2])

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

	def __round__(self, n=None):
		return Vector3(round(self.x, n), round(self.y, n), round(self.z, n))

	def __str__(self):
		return "x: " + str(round(self.x, 3)) + ", y: " + str(round(self.y, 3)) + ", z: " + str(round(self.z, 3))

	def scalar_mul(self, val):
		return Vector3(self.x*val, self.y*val, self.z*val)

	def cross_mul(self, other):
		x = self.y*other.z - self.z*other.y
		y = self.z*other.x - self.x*other.z
		z = self.x*other.y - self.y*other.x
		return Vector3(x, y, z)

	def e_mul(self, other):
		return Vector3(self.x*other.x, self.y*other.y, self.z*other.z)

	def normalize(self):
		return self.scalar_mul(1/abs(self))


class Vector:
	def __init__(self, values):
		self.values = values
		self.size = len(values)

	def as_list(self):
		return self.values[:]

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

	def e_mul(self, other):
		res = Vector(self.values[:])
		for i in range(self.size):
			res.values[i] *= other.values[i]
		return res

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


def rotate_around_vector(p, axis, angle):
	"""
	rotates point p around the axis by the given angle
	:param p: Vector3 (or list of 3 floats) describing a point in 3d space
	:param axis: Vector3 (or list of 3 floats) describing the axis of rotation
	:param angle: the angle by which p will be rotated around the axis (in radians)
	:return: the new position of p
	"""

	if not isinstance(p, Vector3):
		p = Vector3(p[0], p[1], p[2])
	if not isinstance(axis, Vector3):
		axis = Vector3(axis[0], axis[1], axis[2])

	c = math.cos(angle)
	s = math.sin(angle)

	r_matrix = [
		[c + axis.x * (1 - c), axis.x * axis.y * (1 - c) - axis.z * s, axis.x * axis.z * (1 - c) + axis.y * s],
		[axis.x * axis.y * (1 - c) + axis.z * s, c + axis.y * axis.y * (1 - c), axis.y * axis.z * (1 - c) - axis.x * s],
		[axis.x * axis.z * (1 - c) - axis.y * s, axis.y * axis.z * (1 - c) + axis.x * s, c + axis.z * axis.z * (1 - c)]
	]

	return np.matmul(r_matrix, p.as_list())


def convert_to_basis(p, basis):
	"""
	:param p: a point in standard coordinates
	:param basis: three Vector3s representing the three basis vectors for the new basis
	:return: point p in relative coordinates to the basis
	"""

	conversion_matrix = np.linalg.inv(basis)
	return np.matmul(conversion_matrix, p.as_list())


def rot_euler(v, xyz):
	"""
	Rotate vector v (or array of vectors) by the euler angles xyz
	https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
	:param v:
	:param xyz: euler angels; tuple of length 3
	:return:
	"""

	for theta, axis in zip(xyz, np.eye(3)):
		v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis * -theta)))
	return v
