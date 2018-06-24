import re
import math
from configparser import ConfigParser
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from time import time

# --------------------Constants--------------------
# naming/paths
SAVE_PATH = "./saved/"
SESSION_NAME = "default"
LOGFILE = "../util/log.txt"
TENSORBOARD_LOG_DIR = "./Tensorboard_Log/"

# nn-format/learning-vars
BATCH_SIZE = 2000  # number of features in a set
N_EPOCHS = 5  # one epoch contains every feature in mnist.train
LEARNING_RATE = 0.1  # size of change the optimizer makes to the variables

# CapsNet
DEFAULT_REPETITIONS = 3

# info
UPDATE_INTERVAL = 1.0  # time between updates during learning in seconds
VERBOSE = True


class NeuralNetwork:
	def __init__(self, n_input, n_output, net_id, do_act_summaries=False, save_path=SAVE_PATH, log_dir=TENSORBOARD_LOG_DIR, gpu_enable=True, input_summary=None):
		# !WARNING: creating an instance of this class will automatically corrupt every existing instance
		tf.reset_default_graph()

		self.net_config = ConfigParser()
		self.net_config["Format"] = {"n_input": str(n_input), "n_output": str(n_output), "n_layers": 0}
		self.net_config["Options"] = {"gpu_enable": str(gpu_enable), "save_path": save_path, "log_dir": log_dir, "in_summary": list_as_string(input_summary), "do_act_summaries": do_act_summaries}
		self.net_config["Stats"] = {"total_steps": 0, "total_time": 0, "epochs": 0, "batches": 0}

		self.do_act_summaries = do_act_summaries		# create activation summaries
		self.save_path = save_path						# directory the net is saved to
		self.log_dir = log_dir							# directory logging data is written to
		self.n_input = n_input							# number of input values (size of the input tensor)
		self.n_output = n_output						# number classes (size of the output tensor)
		self.id = net_id								# id the net is identified by

		self.x = tf.placeholder(tf.float32, shape=[None, n_input], name="x")
		self.labels = tf.placeholder(tf.float32, shape=[None, n_output], name="labels")
		self.prediction = self.x

		# input summary (as image) can be added if size fits
		if input_summary is not None and len(input_summary) == 2 and input_summary[0] * input_summary[1] == n_input:
			tf.summary.image("input", tf.reshape(self.x, [-1, input_summary[0], input_summary[1], 1]))

		self.committed = False
		self.current_out = [n_input]					# the format of the current output tensor
		self.name_tracker = {}							# tracks variable names to avoid duplicate names
		self.activations = {"relu": tf.nn.relu,
							"sigmoid": tf.nn.sigmoid}
		self.pooling = {"avg": tf.nn.avg_pool,
						"max": tf.nn.max_pool}

		# tf_config to allow dynamic vram allocation to avoid out of memory errors
		self.tf_config = tf.ConfigProto()
		self.tf_config.gpu_options.allow_growth = True
		if not gpu_enable:
			self.tf_config.device_count["GPU"] = 0
		self.session = tf.Session(config=self.tf_config)
		self.saver = None								# the saver is created after the net is committed

		self.format = net_id + "[in_" + str(n_input)	# string of hyperparameters to be distinguished in tensorboard

		self.train_step = None							# the operation to be run in every training step; created when training is set up
		self.merged_summary = None						# a merged summary of all summaries contained in this net; created when all summaries are created
		self.accuracy = None							# operation to be run in every testing step; created when training is set up

		if VERBOSE:
			print(net_id, "was initialized")

	@staticmethod
	def restore(save_dir, net_id, new_id=None):
		"""
		restores the graph, variables and information of a saved net from file system
		:param save_dir: the directory the net is saved in
		:param net_id: the id of the saved net
		:param new_id: optional new id (if a new name is given the saved net is not overridden)
		:return:
		"""

		config = ConfigParser()
		config.read(save_dir + net_id + "/net_config.ini")

		if new_id is not None:
			net_name = new_id
		else:
			net_name = net_id

		try:
			log_dir = config["Options"]["log_dir"]
		except KeyError:
			log_dir = TENSORBOARD_LOG_DIR

		net = NeuralNetwork(int(config["Format"]["n_input"]), int(config["Format"]["n_output"]), net_name, do_act_summaries=bool(config["Options"]["do_act_summaries"]), save_path=save_dir, log_dir=log_dir, input_summary=int_list_from_string(config["Options"]["in_summary"]), gpu_enable=bool(config["Options"]["gpu_enable"]))
		for l_index in range(int(config["Format"]["n_layers"])):
			l_type = config["layer_" + str(l_index)]["type"]
			if l_type == "fc":
				net.fully_connected_layer(int(config["layer_" + str(l_index)]["n_nodes"]),
											img_summ=int_list_from_string(config["layer_" + str(l_index)]["summary"]))
			elif l_type == "rec":
				net.recurrent_layer(int(config["layer_" + str(l_index)]["n_nodes"]))
			elif l_type == "conv":
				net.convolutional_layer(int(config["layer_" + str(l_index)]["width"]),
										int(config["layer_" + str(l_index)]["height"]),
										strides=int_list_from_string(config["layer_" + str(l_index)]["strides"]),
										padding=config["layer_" + str(l_index)]["padding"],
										in_channels=int(config["layer_" + str(l_index)]["in_channels"]),
										out_channels=int(config["layer_" + str(l_index)]["out_channels"]),
										filter_width=int(config["layer_" + str(l_index)]["filter_width"]),
										filter_height=int(config["layer_" + str(l_index)]["filter_height"]))
			elif l_type == "pool":
				net.pool(config["layer_" + str(l_index)]["pool_id"],
							k_size=int_list_from_string(config["layer_" + str(l_index)]["k_size"]),
							strides=int_list_from_string(config["layer_" + str(l_index)]["strides"]),
							padding=config["layer_" + str(l_index)]["padding"])
			elif l_type == "act":
				net.activation(config["layer_" + str(l_index)]["act_type"])
			elif l_type == "dout":
				net.dropout(float(config["layer_" + str(l_index)]["act_type"]),
							noise_shape=int_list_from_string(config["layer_" + str(l_index)]["noise_shape"]))
			elif l_type == "r_shape":
				net.reshape(int_list_from_string(config["layer_" + str(l_index)]["shape"]))
			else:
				raise ValueError

		net.commit()
		net.add_train_setup(float(config["Options"]["learning_rate"]))
		net.load_vars(save_path=save_dir + net_id + "/" + net_id + ".ckpt")

		for key in config["Stats"]:
			net.net_config["Stats"][key] = config["Stats"][key]

		return net

	@staticmethod
	def convert(save_dir, net_id, enable_gpu=True):
		"""
		converts a not of an old saving format to a new net
		:param save_dir: directory the net is saved in
		:param net_id: the id of the saved net
		:param enable_gpu: whether gpu acceleration is enabled for this net
		:return:
		"""

		with open(save_dir + net_id + "/" + "format.txt", "r") as format_file:
			net_format = format_file.readline()
		layers = re.search("\[(.*?)\]", net_format).group(1).split("-")		# "-" conflicts with possible "-1" in reshape vars
		net_id = net_format.split("[")[0]
		n_in = layers[0].split("_")[1]
		n_out = layers[-1].split("_")[1]

		ret_net = NeuralNetwork(int(n_in), int(n_out), net_id, gpu_enable=enable_gpu)

		for layer_str in layers[1:-1]:
			parts = layer_str.split("_")
			if len(parts) == 2:
				f_code, param = parts
			else:
				print("read invalid function description")

			if f_code == "fc":
				ret_net.fully_connected_layer(int(param))
			elif f_code == "rec":
				ret_net.recurrent_layer(int(param))
			elif f_code == "conv":
				height, width = param.split(",")
				ret_net.convolutional_layer(int(height), int(width))
			elif f_code == "pool":
				ret_net.pool(param)
			elif f_code == "act":
				ret_net.activation(param)
			elif f_code == "rs":
				shape = []
				for s in param.split(","):
					shape.append(int(s))
				ret_net.reshape(shape)

		ret_net.commit()
		reg_res = re.search("lr=[0-9]\.[0-9]+", net_format).group(0)
		lr = float(reg_res.lstrip("lr="))
		ret_net.add_train_setup(learning_rate=lr)

		with tf.Session(config=ret_net.tf_config) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, save_dir + net_id + "/" + net_id + ".ckpt")

		if VERBOSE:
			print("restored", ret_net.id)

		return ret_net

	def fully_connected_layer(self, n_nodes, img_summ=None):
		"""
		adds a fully connected layer to the net
		:param n_nodes: number of neurons in the layer
		:param img_summ: whether an image-summary is created for this layers output
		:return:
		"""

		if self.committed:
			print("Net is committed and can not be changed")
			return
		if len(self.current_out) != 1:
			in_nodes = 1
			for l in self.current_out:
				in_nodes *= l
			self.prediction = tf.reshape(self.prediction, shape=[-1, in_nodes], name="fc_rs")
			self.current_out = [in_nodes]
			print("reshaped input - fc-layers require single dimension input")

		try:
			self.name_tracker["fc"] += 1
		except KeyError:
			self.name_tracker["fc"] = 0

		with tf.name_scope("fc-layer"), tf.variable_scope("fc-layer" + str(self.name_tracker["fc"])):
			# weights = tf.Variable(tf.random_normal([self.current_out[0], n_nodes]), name="w")
			weights = tf.get_variable("w", shape=[self.current_out[0], n_nodes], initializer=tf.glorot_normal_initializer())
			biases = tf.Variable(tf.zeros([n_nodes]), name="b")
			tf.summary.histogram("weights", weights)
			tf.summary.histogram("biases", biases)

			self.prediction = tf.matmul(self.prediction, weights) + biases
			if img_summ is not None and len(img_summ) == 2 and img_summ[0]*img_summ[1] == n_nodes:
				fc_summ_img = tf.reshape(self.prediction, [-1, img_summ[0], img_summ[1], 1], "fc_is")
				tf.summary.image("fc_out", fc_summ_img)

			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		self.current_out = [n_nodes]
		self.format += "-fc_" + str(n_nodes)

		n_layers = int(self.net_config["Format"]["n_layers"])
		self.net_config["layer_" + str(n_layers)] = {"type": "fc", "n_nodes": str(n_nodes), "summary": list_as_string(img_summ)}
		self.net_config["Format"]["n_layers"] = str(n_layers + 1)

		if VERBOSE:
			print("added fully connected layer to", self.id)

	def recurrent_layer(self, n_nodes):
		"""
		adds a recurrent layer to the net
		:param n_nodes: number of LSTM-Cells in this layer
		:return:
		"""

		if self.committed:
			print("Net is committed and can not be changed")
			return
		if len(self.current_out) != 1:
			in_nodes = 1
			for l in self.current_out:
				in_nodes *= l
			self.prediction = tf.reshape(self.prediction, shape=[-1, in_nodes], name="rec_rs")
			self.current_out = [in_nodes]
			print("reshaped input - rnn-layers require single dimension input")

		try:
			self.name_tracker["rec"] += 1
		except KeyError:
			self.name_tracker["rec"] = 0

		with tf.name_scope("rec-layer"), tf.variable_scope("rec-layer" + str(self.name_tracker["rec"])):
			# weights = tf.Variable(tf.random_normal([n_nodes, n_nodes]), name="w")
			weights = tf.get_variable("w", shape=[n_nodes, n_nodes], initializer=tf.glorot_normal_initializer())
			biases = tf.Variable(tf.zeros([n_nodes]), name="b")

			# returns current_out[0] number of tensors
			self.prediction = tf.split(self.prediction, self.current_out[0], 1)

			rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_nodes)])
			outputs, states = rnn.static_rnn(rnn_cell, self.prediction, dtype=tf.float32)

			self.prediction = tf.matmul(outputs[-1], weights) + biases

			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		self.current_out = [n_nodes]
		self.format += "-rec_" + str(n_nodes)

		n_layers = int(self.net_config["Format"]["n_layers"])
		self.net_config["layer_" + str(n_layers)] = {"type": "rec", "n_nodes": str(n_nodes)}
		self.net_config["Format"]["n_layers"] = str(n_layers + 1)

		if VERBOSE:
			print("added recurrent layer to", self.id)

	# todo add support for VALID padding
	def convolutional_layer(self, width, height, strides=[1, 1, 1, 1], padding="SAME", in_channels=1, out_channels=1, filter_width=3, filter_height=3):
		"""
		adds a convolutional layer to this net
		:param width: width of the input image
		:param height: height of the input image
		:param strides: see tensorflow documentation
		:param padding: see tensorflow documentation
		:param in_channels: number of channels in the input image
		:param out_channels: number of channels in the output image
		:param filter_width: with of the filter
		:param filter_height: height of the filter
		:return:
		"""

		if self.committed:
			print("Net is committed and can not be changed")
			return
		if not (padding == "SAME" or padding == "VALID"):
			print("padding type not recognized - must be SAME or VALID")
			return

		in_prod = 1
		for l in self.current_out:
			in_prod *= l
		if width*height*in_channels != in_prod:
			print("requested format not compatible; req:", [height, width, in_channels], "->", width*height*in_channels, "is:", in_prod)
			return
		if width*height*out_channels != in_prod:
			print("requested out channels are not compatible - default to keep channels")
			out_channels = in_channels

		if len(self.current_out) != 3 or in_channels != self.current_out[2]:
			self.prediction = tf.reshape(self.prediction, [-1, height, width, in_channels], name="conv_rs")
			print("reshaped input - mismatching input for convolution")
		else:
			height = self.current_out[0]
			width = self.current_out[1]

		try:
			self.name_tracker["conv"] += 1
		except KeyError:
			self.name_tracker["conv"] = 0

		with tf.name_scope("conv-layer"), tf.variable_scope("conv-layer", str(self.name_tracker["conv"])):
			tf.summary.image("conv_in", self.prediction)

			# conv_filter = tf.Variable(tf.random_normal([filter_height, filter_width, in_channels, out_channels]), name="conv_filter")
			conv_filter = tf.get_variable("conv_filter", shape=[filter_height, filter_width, in_channels, out_channels], initializer=tf.glorot_normal_initializer())
			tf.summary.histogram("filter", conv_filter)
			self.prediction = tf.nn.conv2d(self.prediction, conv_filter, strides, padding=padding)

			tf.summary.image("conv_out", self.prediction)
			if padding == "SAME":
				out_height = int(math.ceil(height/strides[1]))
				out_width = int(math.ceil(width/strides[2]))
			elif padding == "VALID":
				out_height = int(math.ceil((height - filter_height + 1) / strides[1]))
				out_width = int(math.ceil((width - filter_width + 1) / strides[2]))

			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		self.current_out = [out_height, out_width, out_channels]
		self.format += "-conv_" + str([height, width]).replace("[", "").replace("]", "").replace(" ", "")
		print("conv output shape:", self.current_out)

		n_layers = int(self.net_config["Format"]["n_layers"])
		self.net_config["layer_" + str(n_layers)] = {
			"type": "conv",
			"width": str(width),
			"height": str(height),
			"strides": list_as_string(strides),
			"padding": str(padding),
			"in_channels": str(in_channels),
			"out_channels": str(out_channels),
			"filter_width": str(filter_width),
			"filter_height": str(filter_height)
		}
		self.net_config["Format"]["n_layers"] = str(n_layers + 1)

		if VERBOSE:
			print("added convolutional layer to", self.id)

	def pool(self, pool_id, k_size=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME"):
		"""
		adds a pooling layer to this net
		:param pool_id: type of pooling function to be used
		:param k_size: see tensorflow documentation
		:param strides: see tensorflow documentation
		:param padding: see tensorflow documentation
		:return:
		"""

		if self.committed:
			print("Net is committed and can not be changed")
			return
		if not (padding == "SAME" or padding == "VALID"):
			print("padding type not recognized - must be SAME or VALID")
			return
		if len(self.current_out) != 3:
			print("pooling requires input: [height, width, channels]")
			return

		with tf.name_scope(pool_id + "_pool"):
			self.prediction = self.pooling[pool_id](self.prediction, k_size, strides, "SAME")
			tf.summary.image("pooled", self.prediction)
			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		if padding == "SAME":
			out_height = int(math.ceil(self.current_out[0] / strides[1]))
			out_width = int(math.ceil(self.current_out[1] / strides[2]))
		elif padding == "VALID":
			out_height = int(math.ceil((self.current_out[0] - k_size[1] + 1) / strides[1]))
			out_width = int(math.ceil((self.current_out[1] - k_size[2] + 1) / strides[2]))

		self.current_out = [out_height, out_width, self.current_out[2]]
		self.format += "-pool_" + pool_id

		n_layers = int(self.net_config["Format"]["n_layers"])
		self.net_config["layer_" + str(n_layers)] = {"type": "pool", "pool_id": pool_id, "k_size": list_as_string(k_size), "strides": list_as_string(strides), "padding": str(padding)}
		self.net_config["Format"]["n_layers"] = str(n_layers + 1)

		if VERBOSE:
			print("added", pool_id, "pool layer to", self.id)

	def activation(self, act_id):
		"""
		adds an activation to the current output of the net
		:param act_id: type of activation function to be used
		:return:
		"""

		if self.committed:
			print("Net is committed and can not be changed")
			return

		with tf.name_scope(act_id):
			self.prediction = self.activations[act_id](self.prediction, act_id)
			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		self.format += "-act_" + act_id

		n_layers = int(self.net_config["Format"]["n_layers"])
		self.net_config["layer_" + str(n_layers)] = {"type": "act", "act_type": str(act_id)}
		self.net_config["Format"]["n_layers"] = str(n_layers + 1)

		if VERBOSE:
			print("added activation", act_id, "to", self.id)

	def dropout(self, rate, noise_shape=None):
		"""
		adds dropout to the current output of this net
		:param rate: percentage of values to be dropped out
		:param noise_shape: see tensorflow documentation
		:return:
		"""

		if self.committed:
			print("Net is committed and can not be changed")
			return

		with tf.name_scope("dout"):
			self.prediction = tf.nn.dropout(self.prediction, 1-rate, noise_shape=noise_shape)
			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		self.format += "-dout_" + str(rate)

		n_layers = int(self.net_config["Format"]["n_layers"])
		self.net_config["layer_" + str(n_layers)] = {"type": "dout", "rate": str(rate), "noise_shape": list_as_string(noise_shape)}
		self.net_config["Format"]["n_layers"] = str(n_layers + 1)

		if VERBOSE:
			print("added", rate, "dropout to", self.id)

	def reshape(self, shape):
		"""
		changes the shape of the current output of this net
		:param shape: the new shape; see tensorflow documentation
		:return:
		"""

		with tf.name_scope("reshape"):
			self.prediction = tf.reshape(self.prediction, shape, "reshape")
			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		self.current_out = shape[1:]
		self.format += "-rs_" + str(shape).replace("[", "").replace("]", "").replace(" ", "")

		n_layers = int(self.net_config["Format"]["n_layers"])
		self.net_config["layer_" + str(n_layers)] = {"type": "r_shape", "shape": list_as_string(shape)}
		self.net_config["Format"]["n_layers"] = str(n_layers + 1)

	def commit(self):
		"""
		disallows further changes to the structure of this net and adds output layer with the previously
		specified number of classes
		:return:
		"""

		if self.committed:
			print("Net is already committed")
			return

		with tf.name_scope("out"), tf.variable_scope("out"):
			if len(self.current_out) != 1:
				in_nodes = 1
				for l in self.current_out:
					in_nodes *= l
				self.prediction = tf.reshape(self.prediction, shape=[-1, in_nodes])
				self.current_out = [in_nodes]
				print("reshaped input - out-layer requires single dimension input")

			# weights = tf.Variable(tf.random_normal([self.current_out[0], self.n_output]), name="w")
			weights = tf.get_variable("w", shape=[self.current_out[0], self.n_output], initializer=tf.glorot_normal_initializer())
			biases = tf.Variable(tf.zeros([self.n_output]), name="b")
			tf.summary.histogram("weights", weights)
			tf.summary.histogram("biases", biases)

			self.prediction = tf.matmul(self.prediction, weights) + biases
			if self.do_act_summaries:
				tf.summary.histogram("activations", self.prediction)

		self.committed = True
		self.current_out = [self.n_output]
		self.format += "-out_" + str(self.n_output) + "]"

		if VERBOSE:
			print(self.id, "was committed with format:", self.format)

	# todo allow the learning rate to change over the course of training
	def add_train_setup(self, learning_rate=LEARNING_RATE):
		"""
		adds all training related operations to the net
		creates the merged summary and the saver
		:param learning_rate: see tensorflow documentation
		:return:
		"""

		# loss
		with tf.name_scope("loss"):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
			tf.summary.scalar("loss", loss)

		# the optimizer determines the way all the variables need to be changed to reduce loss
		# simple gradient descent optimization
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
		with tf.name_scope("train"):
			self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

		with tf.name_scope("test"):
			correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			tf.summary.scalar("accuracy", self.accuracy)

		self.merged_summary = tf.summary.merge_all()
		self.format += ", lr=" + str(learning_rate)

		self.net_config["Options"]["learning_rate"] = str(learning_rate)

		# WARNING - if a net is not given this training setup its variables are never initialized nor saved
		self.session.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

		if VERBOSE:
			print("training set up for", self.id)

	def train(self, training_data, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, do_save=True, save_path=None, auto_load=False):
		"""
		runs the training with the provided data
		:param training_data: the data the net is trained on (list of inputs and labels)
		:param n_epochs: number of epochs (one epoch is one complete iteration through the data)
		:param batch_size: size of the batches of data run in every iteration
		:param do_save: whether the net is saved at the end of training
		:param save_path: where the net is saved to
		:param auto_load: automatically try to load saved variables
		:return:
		"""

		if save_path is None:
			save_path = self.save_path
		if self.net_config["Stats"]["total_steps"] == "0":
			self.format += ", ep=" + str(n_epochs)
			self.format += ", bs=" + str(batch_size)
			self.format += ", gpu=" + "yes" if self.net_config["Options"]["gpu_enable"] == "True" else "no"

		if auto_load:
			self.load_vars()

		print("Data set size:", len(training_data))

		file_writer = tf.summary.FileWriter(self.log_dir + self.format)
		file_writer.add_graph(self.session.graph)
		print("starting...")

		start_time = time()
		prev_time = start_time  # keeps track of the time the last progress-update was printed
		steps = int(self.net_config["Stats"]["total_steps"])
		print("training from step", steps)

		for epoch in range(n_epochs):
			for batch_index in range(int(math.ceil(len(training_data) / batch_size))):
				start = batch_index * batch_size
				end = start + batch_size

				batch_x = np.array([i[0] for i in training_data[start:end]])
				batch_y = np.array([i[1] for i in training_data[start:end]])

				# print("bx:", batch_x.shape)
				# print("by:", batch_y.shape)

				_, batch_summary = self.session.run([self.train_step, self.merged_summary], feed_dict={self.x: batch_x, self.labels: batch_y})
				file_writer.add_summary(batch_summary, steps)
				steps += 1

				# print the progress in intervals of specified length
				curr_time = time()
				if curr_time - prev_time > UPDATE_INTERVAL:
					prev_time = curr_time
					epoch_progress = round(batch_index / (len(training_data) / batch_size), 4)
					total_progress = round((epoch + epoch_progress) / n_epochs, 4)

					# example: "Epoch 3/10 : 82.91% | 38.29% (epoch|total)"
					print("Epoch", str(epoch + 1) + "/" + str(n_epochs) + ":", str(round(epoch_progress * 100, 2)) + "%", "|", str(round(total_progress * 100, 2)) + "%", "(epoch|total)")

		end_time = time()

		self.net_config["Stats"]["epochs"] = str(int(self.net_config["Stats"]["epochs"]) + n_epochs)
		self.net_config["Stats"]["batches"] = str(int(self.net_config["Stats"]["batches"]) + math.ceil(len(training_data) / batch_size))
		self.net_config["Stats"]["total_time"] = str(int(self.net_config["Stats"]["total_time"]) + int(end_time - start_time))
		self.net_config["Stats"]["total_steps"] = str(steps)

		# save the session to file
		if do_save:
			path = save_path + self.id + "/"
			sess_save_path = path + self.id + ".ckpt"
			self.saver.save(self.session, sess_save_path)
			with open(path + "net_config.ini", "w") as config_file:
				self.net_config.write(config_file)
			print("saved net to: ", path)

	def test(self, testing_data, batch_size=BATCH_SIZE):
		"""
		test the accuracy of the net on the provided data without training
		:param testing_data: the data to be used for testing
		:param batch_size: size of the batches run in every iteration
		:return:
		"""

		accuracy = -1

		for batch_index in range(int(math.ceil(len(testing_data) / batch_size))):
			start = batch_index * batch_size
			end = start + batch_size

			batch_x = np.array([i[0] for i in testing_data[start:end]])
			batch_y = np.array([i[1] for i in testing_data[start:end]])

			accuracy = self.session.run([self.accuracy], feed_dict={self.x: batch_x, self.labels: batch_y})[0]

		return round(100 * accuracy, 2)

	def load_vars(self, save_path=None):
		"""
		loads saved variables
		:param save_path: directory the variables are saved in
		:return:
		"""

		if save_path is None:
			save_path = self.save_path + self.id + "/" + self.id + ".ckpt"

		print("loading variables from:", save_path)
		try:
			if tf.train.checkpoint_exists(save_path):
				self.saver.restore(self.session, save_path)
			else:
				print("no checkpoint found")
		except tf.python.framework.errors_impl.NotFoundError:
			print("no checkpoint found - err")

	# todo only works correct for one hot output
	def predict(self, input_tensor, verbose=VERBOSE):
		"""
		gives the nets output for a given input
		:param input_tensor: the input for net net
		:param verbose: whether additional information is given
		:return:
		"""

		prediction = self.session.run([self.prediction], feed_dict={self.x: [input_tensor]})[0][0]  # double zero index for 1.Run return tuple index and 2.Batch index
		chosen_class = np.argmax(prediction)
		certainty = prediction[chosen_class]

		if verbose:
			print("class=" + str(chosen_class) + ", cert=" + str(certainty))
		return chosen_class, certainty


def list_as_string(l):
	"""
	creates a string from a list that is easily turned back into a list
	:param l: list
	:return:
	"""

	return str(l).replace("[", "").replace("]", "").replace(", ", ":")


def int_list_from_string(s):
	"""
	creates a list of integers from a string (counterpart to list_as_string(l))
	:param s: string
	:return:
	"""

	if s == "None":
		return None
	l = []
	for element in s.split(":"):
		l.append(int(element))
	return l
