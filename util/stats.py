import statistics


class DistributionInfo:
	def __init__(self, data):
		self.data = data
		self.size = len(data)
		if self.size == 1:
			self.mean = data[0]
			self.std_dev = 0
		else:
			self.mean = statistics.mean(data)
			self.std_dev = statistics.stdev(data)
		self.max = max(data)
		self.min = min(data)

	def __str__(self):
		return "DI(" + str(self.size) + "){" + "mean=" + str(round(self.mean, 3)) + ", std_dev=" + str(round(self.std_dev, 3)) + "}"


def average_into(l, n_points):
	if len(l) <= n_points:
		return l

	res = []
	batch_size = len(l)/n_points
	for i in range(n_points):
		begin = int(i * batch_size)
		if i == n_points:
			end = -1
		else:
			end = int((i+1)*batch_size)
		res.append(sum(l[begin:end]) / (end-begin))

	return res
