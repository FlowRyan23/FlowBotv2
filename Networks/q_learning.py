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
		self.merged_summary = None
		# todo adding file_writer produces error: "TypeError: Expected an event_pb2.Event proto, but got <class 'tensorflow.core.util.event_pb2.Event'>"
		# self.file_writer = tf.summary.FileWriter(LOG_DIR + self.name + "/tb")

		self.committed = False
		self.n_layers = 0

		self.tf_config = tf.ConfigProto()
		self.tf_config.gpu_options.allow_growth = True

	def add_fc(self, size, activation, verbose=False):
		if self.committed:
			if verbose:
				print(self.name, "is already committed. Can not add fc layer")
			return

		if verbose:
			print("adding fc layer to {0:s} with size {1:d} and activation {2:s}".format(self.name, size, ActivationType.string_of(activation)))

		self.output = tf.layers.dense(self.output, units=size,
									activation=activation,
									kernel_initializer=tf.glorot_normal_initializer(),
									name="L"+str(self.n_layers)+"-fc",
									bias_initializer=tf.random_normal_initializer())

		with tf.name_scope("L" + str(self.n_layers)):
			tf.summary.histogram("act", self.output)

		self.net_config["Layer" + str(self.n_layers)] = {"type": "fc", "size": size, "activation": ActivationType.string_of(activation)}
		self.n_layers += 1
		self.net_config["Format"]["n_layers"] = str(self.n_layers)

		if verbose:
			print("Added fc layer to", self.name)

	def add_drop_out(self, rate, verbose=False):
		if self.committed:
			if verbose:
				print(self.name, "is already committed. Can not add drop out layer")
			return

		if verbose:
			print("adding drop out layer to {0:s} with rate {1:d}".format(self.name, rate))

		self.output = tf.layers.dropout(self.output, rate,
									name="L"+str(self.n_layers)+"-do")

		with tf.name_scope("L" + str(self.n_layers)):
			tf.summary.histogram("act", self.output)

		self.net_config["Layer" + str(self.n_layers)] = {"type": "do", "rate": rate}
		self.n_layers += 1
		self.net_config["Format"]["n_layers"] = str(self.n_layers)

		if verbose:
			print("Added drop out layer to", self.name)

	def commit(self, verbose=False):
		if self.committed:
			if verbose:
				print(self.name, "is already committed. Can not commit again")
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
		# print("merging summaries")
		self.merged_summary = tf.summary.merge_all()
		# print("adding graph")
		# self.file_writer.add_graph(self.session.graph)

		if verbose:
			print(self.name, "was successfully committed")

	def close(self):
		self.session.close()
		print(self.name, "was closed")

	def run(self, states):
		return self.session.run(self.output, feed_dict={self.x: states})[0]		# todo remove [0] when more than one operation is run

	def train(self, trainining_data, batch_size, n_epochs, learning_rate=1e-3, save=True):
		steps = 0
		start_time = time()

		n_batches = len(trainining_data[0]) / batch_size
		if int(n_batches) != n_batches:
			n_batches = int(n_batches) + 1

		for epoch in range(n_epochs):
			for i in range(n_batches):
				start = i*batch_size
				end = (i+1)*batch_size if i < n_batches else -1		# the last batch may not be full size, in this case all remaining elements will be chosen
				states_batch = trainining_data[0][start:end]
				q_values_batch = trainining_data[1][start:end]
				steps += len(states_batch)

				feed_dict = {self.x: states_batch,
							self.q_values_new: q_values_batch,
							self.learning_rate: learning_rate}

				_, summary = self.session.run([self.optimizer, self.merged_summary], feed_dict=feed_dict)
				# print("adding summary")
				# self.file_writer.add_summary(summary, global_step=self.get_step_count() + steps)

		self.net_config["Stats"]["total_steps"] = str(int(self.net_config["Stats"]["total_steps"]) + steps)
		self.net_config["Stats"]["total_time"] = str(float(self.net_config["Stats"]["total_time"]) + time() - start_time)
		if save:
			self.save()

	def save(self):
		if not os.path.isdir(self.save_path + self.name):
			os.makedirs(self.save_path + self.name)
		with open(self.save_path + self.name + "/net.cfg", "w") as cfg_file:
			self.net_config.write(cfg_file)
		save_path = self.saver.save(self.session, self.save_path + self.name + "/" + self.name)
		print("saved net to:", save_path)

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

		n_layers = int(config["Format"]["n_layers"])
		print("N_Layers:", n_layers)
		for i in range(n_layers):
			l_type = config["Layer" + str(i)]["type"]
			if l_type == "fc":
				size = int(config["Layer" + str(i)]["size"])
				activation = ActivationType.get(config["Layer" + str(i)]["activation"])
				net.add_fc(size, activation, verbose=verbose)
			elif l_type == "do":
				rate = float(config["Layer" + str(i)]["rate"])
				net.add_drop_out(rate, verbose=verbose)

		for key in config["Stats"]:
			net.net_config["Stats"][key] = config["Stats"][key]

		net.commit()
		net.load(path + name + "/" + name) # + ".ckpt")

		return net

	def get_step_count(self):
		return int(self.net_config["Stats"]["total_steps"])


class ActivationType(Enum):
	RELU = tf.nn.relu
	SIGMOID = tf.nn.sigmoid

	@staticmethod
	def get(act_type):
		if act_type == "RELU":
			return ActivationType.RELU
		if act_type == "SIGMOID":
			return ActivationType.SIGMOID

	@staticmethod
	def string_of(act_type):
		if act_type == ActivationType.RELU:
			return "RELU"
		if act_type == ActivationType.SIGMOID:
			return "SIGMOID"


class ReplayMemory:
	def __init__(self, n_actions, discount_factor=0.97):
		self.n_actions = n_actions
		self.discount_factor = discount_factor

		self.size = 0
		self.states = []
		self.q_values = []		# each entry is a list of n_actions float values
		self.actions = []
		self.rewards = []

		self.predicted_q_values = []

		self.estimation_errors = []

	def add(self, state, q_values, action, reward):
		"""
		adds a new entry to memory
		:param state: the state of the environment
		:param q_values: the predicted q-values for every possible action in this state
		:param action: the action that was chosen
		:param reward: the reward received for the transition from state to next state through performing action
		:return:
		"""
		self.states.append(state)
		self.q_values.append(q_values)
		self.actions.append(action)
		self.rewards.append(reward)
		self.size += 1

	def update_q_values(self, sarsa=False):
		"""
		calculates the actual q-values from the predicted ones. during play only the predicted values are stored
		as the real ones are not known (they depend on future states). ideally this method is called at the end of
		an episode -> no future states exist/have any influence on the value of current or past states.
		:param sarsa: whether the sarsa (state action reward state action) variant of q-value updates should be used
						the sarsa-variant uses the q-value of the selected next action instead of the highest q-value next action
		:return:
		"""
		if self.size < 1:
			return -1

		# save a copy of the predicted values for analysis
		self.predicted_q_values = self.q_values[:]

		start_time = time()
		self.estimation_errors = np.zeros(shape=[self.size])

		# the q-value of the last step should be the reward in that step
		self.q_values[-1][self.actions[-1]] = self.rewards[-1]

		# the update moves from the present to the past -> from the back to the front of the array
		for i in reversed(range(self.size-1)):
			action = self.actions[i]
			reward = self.rewards[i]

			# the q-value for the action is composed of the immediate reward + discounted future reward
			if sarsa:
				following_action = self.actions[i + 1]
				action_value = reward + self.discount_factor * self.q_values[i + 1][following_action]
			else:
				action_value = reward + self.discount_factor * np.max(self.q_values[i+1])
			self.estimation_errors[i] = abs(action_value - self.q_values[i][action])

			# only the q-value for the selected action can be updated
			self.q_values[i][action] = action_value
		end_time = time()

		print("Average estimation error:", Stat(self.estimation_errors))
		return end_time - start_time

	def get_random_batch(self, batch_size, duplicates=False):
		"""
		get a batch of randomly selected state-q_value-pairs
		:param duplicates: whether duplicates are allowed
		:param batch_size: the number of state-q_value-pairs returned; if not enough entries are available
							and not duplicates are allowed less pairs are returned
		:return: a list of states and a list of q-values. corresponding entries have the same index
		"""
		if self.size <= 0:
			return None

		# if the batch size is greater than the size of the memory the entire memory is returned
		if batch_size > self.size-1 and not duplicates:
			return self.states, self.q_values

		selection = np.random.choice([i for i in range(self.size)], size=batch_size, replace=duplicates)
		states_batch = np.array(self.states)[selection]
		q_values_batch = np.array(self.q_values)[selection]

		return states_batch, q_values_batch

	def get_training_set(self, shuffle=False):
		if shuffle:
			ordering = [i for i in range(self.size)]
			np.random.shuffle(ordering)
			states = [self.states[i] for i in ordering]
			q_values = [self.q_values[i] for i in ordering]
		else:
			states = self.states
			q_values = self.q_values

		return states, q_values

	def clear(self):
		"""
		deletes all values from memory. only the number of actions and the discount factor are conserved
		:return:
		"""
		self.size = 0
		self.states = []
		self.q_values = []
		self.actions = []
		self.rewards = []

		self.estimation_errors = []

	def write(self):
		np.savetxt(TEMP_DIR + "estimation_errors.csv", self.estimation_errors, delimiter=",")
		# np.savetxt(TEMP_DIR + "rewards.csv", self.rewards, delimiter=",")		# replaced with lines 187-189
		np.savetxt(TEMP_DIR + "q_values.csv", self.q_values, delimiter=",")
		np.savetxt(TEMP_DIR + "pred_q_values.csv", self.predicted_q_values, delimiter=",")

		with open(TEMP_DIR + "rewards.csv", "a") as rewards_file:
			if len(self.rewards) > 0:
				# avrg_reward = sum(self.rewards) / len(self.rewards)
				rewards_file.write(str(sum(self.rewards)) + "\n")
			else:
				print("No rewards to be written")

		with open(TEMP_DIR + "avrg_estimation_errors.csv", "a") as file:
			if len(self.estimation_errors) > 0:
				avrg_est_err = sum(self.estimation_errors) / len(self.estimation_errors)
				file.write(str(avrg_est_err) + "\n")
			else:
				print("No estimation errors to be written")

		states_differentials = []
		prev_state = self.states[0]
		for state in self.states[1:]:
			differential = [round(state[i] - prev_state[i], 4) for i in range(len(state))]
			states_differentials.append(differential)
			prev_state = state
		np.savetxt(TEMP_DIR + "state_diffs.csv", states_differentials, delimiter=",")