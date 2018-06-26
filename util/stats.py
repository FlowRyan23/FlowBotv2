import statistics
import numpy as np


class DistributionInfo:
	def __init__(self, data):
		self.data = data
		self.size = len(data)
		self.mean = statistics.mean(data)
		self.std_dev = statistics.stdev(data)
		self.max = max(data)
		self.min = min(data)

	def __str__(self):
		return "DistInfo(" + str(self.size) + "){" + "mean=" + str(self.mean) + ", std_dev=" + str(self.std_dev) + "}"


def average_into(l, n_points):
	res = np.array_split(l, n_points)
	for i in range(len(res)):
		res[i] = sum(res[i]) / len(res[i])
	return res
