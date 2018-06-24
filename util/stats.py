import statistics


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
