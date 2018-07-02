import os
import numpy as np
import tensorflow as tf
from enum import Enum
from time import time
from configparser import ConfigParser
from util.stats import DistributionInfo as Stat


TEMP_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/temp/"
LOG_DIR = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/util/logs/"
DEFAULT_SAVE_PATH = "E:/Studium/6. Semester/Bachelorarbeit/Code/RLBotPythonExample/Networks/saved/"


class NeuralNetwork:
	def __init__(self, name, input_shape, n_classes, save_path=DEFAULT_SAVE_PATH):
		self.net_config = ConfigParser()
		self.net_config["Format"] = {"input_shape": str(input_shape).rstrip("]").lstrip("["), "n_classes": str(n_classes), "n_layers": 0}
		self.net_config["Options"] = {"save_path": save_path}
		self.net_config["Stats"] = {"total_steps": 0, "total_time": 0}

		self.name = name
		self.n_classes = n_classes
		self.save_path = save_path

		self.x = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name="x")
		self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
		self.q_values_new = tf.placeholder(tf.float32, shape=[None, n_classes], name='q_values_new')

		self.output = self.x
		self.loss = None
		self.session = None
		self.saver = None
		self.optimizer = None

		self.committed = False
		self.n_layers = 0

		self.tf_config = tf.ConfigProto()
		self.tf_config.gpu_options.allow_growth = True

	def add_fc(self, size, activation):
		if self.committed:
			return

		self.output = tf.layers.dense(self.output, units=size,
									activation=activation,
									kernel_initializer=tf.glorot_normal_initializer(),
									name="L"+str(self.n_layers)+"-fc",
									bias_initializer=tf.random_normal_initializer())

		self.net_config["Layer" + str(self.n_layers)] = {"type": "fc", "size": size, "activation": activation}
		self.n_layers += 1
		self.net_config["Format"]["n_layers"] = str(self.n_layers)

	def add_drop_out(self, rate):
		if self.committed:
			return

		self.output = tf.layers.dropout(self.output, rate,
									name="L"+str(self.n_layers)+"-do")

		self.net_config["Layer" + str(self.n_layers)] = {"type": "do", "rate": rate}
		self.n_layers += 1
		self.net_config["Format"]["n_layers"] = str(self.n_layers)

	def commit(self):
		if self.committed:
			return

		self.output = tf.layers.dense(self.output, units=self.n_classes,
									activation=ActivationType.RELU,
									kernel_initializer=tf.glorot_normal_initializer(),
									name="out",
									bias_initializer=tf.random_normal_initializer())

		squared_error = tf.square(self.output - self.q_values_new)
		sum_squared_error = tf.reduce_sum(squared_error, axis=1)
		self.loss = tf.reduce_mean(sum_squared_error)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		self.saver = tf.train.Saver()
		self.session = tf.Session(config=self.tf_config)
		self.session.run(tf.global_variables_initializer())

	def close(self):
		self.session.close()

	def run(self, states):
		return self.session.run(self.output, feed_dict={self.x: states})[0]		# todo remove [0] when more than one operation is run

	def train(self, replay_memory, batch_size, n_epochs, learning_rate=0.01, save=True):
		steps = 0
		start_time = time()

		# todo replay memory may return less than batch_size elements when ist does not contain enough -> adjust training length
		n_batches = replay_memory.size / batch_size
		n_batches *= n_epochs
		n_batches = int(n_batches) + 1
		for i in range(n_batches):
			states_batch, q_values_batch = replay_memory.get_random_batch(batch_size=batch_size)
			steps += len(states_batch)

			feed_dict = {self.x: states_batch,
						self.q_values_new: q_values_batch,
						self.learning_rate: learning_rate}

			self.session.run(self.optimizer, feed_dict=feed_dict)

		self.net_config["Stats"]["total_steps"] = str(int(self.net_config["Stats"]["total_steps"]) + steps)
		self.net_config["Stats"]["total_time"] = str(float(self.net_config["Stats"]["total_time"]) + time() - start_time)
		if save:
			self.save()

	def save(self):
		if not os.path.isdir(self.save_path + self.name):
			os.makedirs(self.save_path + self.name)
		with open(self.save_path + self.name + "/net.cfg", "w") as cfg_file:
			self.net_config.write(cfg_file)
		self.saver.save(self.session, self.save_path + self.name + "/" + self.name)

	def load(self, ckp_file):
		self.saver.restore(self.session, save_path=ckp_file)

	@staticmethod
	def restore(name, path, new_name=None, verbose=False):
		if verbose:
			print("restoring {0:s} from {1:s}".format(name, path))
			print("config file:", path + name + "/net.cfg")
			print("tf checkpoint:", path + name + "/" + name + ".ckpt")

		config = ConfigParser()
		config.read(path + name + "/net.cfg")

		name_ = (name if new_name is None else new_name)
		input_shape = [int(s) for s in config["Format"]["input_shape"].split(",")]
		n_classes = int(config["Format"]["n_classes"])
		net = NeuralNetwork(name_, input_shape, n_classes, save_path=path)

		for i in range(int(config["Format"]["n_layers"])):
			l_type = config["Layer" + str(i)]
			if l_type == "fc":
				size = int(config["Layer" + str(i)]["size"])
				activation = ActivationType.get(config["Layer" + str(i)]["activation"])
				net.add_fc(size, activation)
			elif l_type == "do":
				rate = float(config["Layer" + str(i)]["rate"])
				net.add_drop_out(rate)

		net.commit()
		net.load(path + name + "/" + name + ".ckpt")


class ActivationType(Enum):
	RELU = tf.nn.relu
	SIGMOID = tf.nn.sigmoid

	def get(self, act_type):
		if act_type == "RELU":
			return self.RELU
		if act_type == "SIGMOID":
			return self.SIGMOID


class ReplayMemory:
	def __init__(self, n_actions, discount_factor=0.97):
		self.n_actions = n_actions
		self.discount_factor = discount_factor

		self.size = 0
		self.states = []
		self.q_values = []		# each entry is a list of n_actions float values
		self.actions = []
		self.rewards = []

		self.estimation_errors = []

	def add(self, state, q_values, action, reward):
		self.states.append(state)
		self.q_values.append(q_values)
		self.actions.append(action)
		self.rewards.append(reward)
		self.size += 1

	def update_q_values(self):
		if self.size < 1:
			return -1

		start_time = time()
		self.estimation_errors = np.zeros(shape=[self.size])
		self.q_values[-1][self.actions[-1]] = self.rewards[-1]
		for i in reversed(range(self.size-1)):
			action = self.actions[i]
			reward = self.rewards[i]

			action_value = reward + self.discount_factor * np.max(self.q_values[i+1])
			self.estimation_errors[i] = abs(action_value - self.q_values[i][action])
			self.q_values[i][action] = action_value
		end_time = time()

		print("Average estimation error:", Stat(self.estimation_errors))
		return end_time - start_time

	def get_random_batch(self, batch_size):
		if self.size <= 0:
			return None

		# if the batch size is greater than the size of the memory the entire memory is returned
		if batch_size > self.size-1:
			return self.states, self.q_values

		selection = np.random.choice([i for i in range(self.size)], size=batch_size, replace=False)
		states_batch = np.array(self.states)[selection]
		q_values_batch = np.array(self.q_values)[selection]

		return states_batch, q_values_batch

	def clear(self):
		self.size = 0
		self.states = []
		self.q_values = []
		self.actions = []
		self.rewards = []

		self.estimation_errors = []

	def write(self):
		np.savetxt(TEMP_DIR + "estimation_errors.csv", self.estimation_errors, delimiter=",")
		np.savetxt(TEMP_DIR + "rewards.csv", self.rewards, delimiter=",")
		np.savetxt(TEMP_DIR + "q_values.csv", self.q_values, delimiter=",")

		states_differentials = []
		prev_state = self.states[0]
		for state in self.states[1:]:
			differential = [round(state[i] - prev_state[i], 4) for i in range(len(state))]
			states_differentials.append(differential)
			prev_state = state
		np.savetxt(TEMP_DIR + "state_diffs.csv", states_differentials, delimiter=",")