
# parses the dataset
import ptb_reader

import tensorflow as tf
import numpy as np
import datetime
import os.path


#import tensorflow.contrib.rnn.LSTMCell as tf_LSTM_Cell
# import tensorflow.nn.rnn_cell.BasicLSTMCell as tf_LSTM_Cell
# tf.nn.rnn_cell.BasicLSTMCell
#from tf.nn.rnn_cell import BasicLSTMCell as tf_LSTM_Cell
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell as tf_LSTM_Cell
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.seq2seq import sequence_loss_by_example


class Neural_Network:

	#Static Member Variables of the Class
	TRAINING_MODE = 1
	TEST_OR_VALIDATION_MODE = 2

	def __init__(self, model_name, num_time_steps, vocab_size, hidden_layer_size, num_hidden_layers, init_scale_half_range):

		self.num_hidden_layers = num_hidden_layers
		self.model_name = model_name
		self.num_time_steps = num_time_steps
		self.vocab_size = vocab_size
		self.hidden_layer_size = hidden_layer_size
		self.init_scale_half_range = init_scale_half_range

		self.batch_counter = 0	# used in next_batch()


	def init_tensorflow_members(self, learning_rate, maximum_gradient_norm):	# operating_mode = Neural_Network.TEST_OR_VALIDATION_MODE
		

		#self.operating_mode = operating_mode
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_probability")

		self.learning_rate = learning_rate
		self.max_grad_norm = maximum_gradient_norm

		self.input_index_sequence = tf.placeholder(tf.int32, [None, self.num_time_steps], name="inputs")
		# this represents the input sequence of integers (indexed words) where say 36452 is the 36453th Word in accumulated Vocabulary 

		self.target_scores = tf.placeholder(tf.int32, [None, self.num_time_steps], name="targets")

		self.var_initializer = var_initializer = tf.random_uniform_initializer(-1.0 * self.init_scale_half_range, +1.0 * self.init_scale_half_range)
		

		with tf.device("/cpu:0"):

			embedding_matrix = tf.get_variable("embedding", [self.vocab_size, self.hidden_layer_size], initializer=var_initializer)
			
			self.input_vector_sequence = tf.nn.embedding_lookup(embedding_matrix, self.input_index_sequence)
			# this is the same sequence of words, but this time, each of the words represented by its respective word vector of size = hidden_layer_size

		#if self.operating_mode == Neural_Network.TRAINING_MODE:
			# If Training right Now, then introduce dropout

		self.input_vector_sequence = tf.nn.dropout(self.input_vector_sequence, self.dropout_keep_prob)

		self.inferred_batch_size = tf.shape(self.input_index_sequence)[0]


		self.init_LSTM_Cells()

		final_LSTM_state, lstm_outputs = self.feed_forward_pass(self.input_vector_sequence)
		feed_forward_output = tf.reshape(tf.concat(1, lstm_outputs), [-1, self.hidden_layer_size])

		sl_w = tf.get_variable("softmax_layer_w", [self.hidden_layer_size, self.vocab_size], initializer=var_initializer)
		sl_b = tf.get_variable("softmax_layer_b", [self.vocab_size], initializer=var_initializer)


		logits = tf.nn.xw_plus_b(feed_forward_output, sl_w, sl_b) # compute logits for loss
		targets = tf.reshape(self.target_scores, [-1]) # reshape our target outputs

		

		loss_scaling_weights = tf.ones([self.inferred_batch_size * self.num_time_steps]) # used to scale the loss average

		# computes loss and performs softmax on our fully-connected output layer
		loss = sequence_loss_by_example([logits], [targets], [loss_scaling_weights], self.vocab_size)
		self.cost = tf.div(tf.reduce_sum(loss),  tf.to_float(self.inferred_batch_size), name="cost")

		
		self.back_propagation_pass()


		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		self.init = tf.initialize_all_variables()


		#saved_index_file_name = "weights/"+self.model_name+".ckpt.index"
		saved_meta_file_name = "weights/"+self.model_name+".ckpt.meta"


		if os.path.exists(saved_meta_file_name): # and os.path.exists(saved_index_file_name):
			print('\n Restoring Model from last checkpoint ... \n\n')
			# file exists, so continue on the saved model
			self.saver.restore(self.sess, "weights/"+self.model_name+".ckpt")
		else:
			print('\n Beginning Model Training from scratch ... \n\n')
			self.sess.run(self.init) # reset values to incorrect defaults.
		


	def init_LSTM_Cells(self):


		#####################################################
		#		NOW DEFINING THE LSTM / RNN PART			#
		#####################################################


		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer_size)

		#if self.operating_mode == Neural_Network.TRAINING_MODE and self.dropout_keep_prob < 1:
		lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
		
		self.multi_layer_Cell = rnn_cell.MultiRNNCell([lstm_cell] * self.num_hidden_layers)
		self.initial_LSTM_state = self.multi_layer_Cell.zero_state(self.inferred_batch_size, tf.float32)



	def feed_forward_pass(self, _input_):

		lstm_outputs = []
		lstm_states = []

		# define the current LSTM state to be the initial one here :
		curr_lstm_state = self.initial_LSTM_state

		with tf.variable_scope("LSTM_feed_Fwd", initializer= self.var_initializer):

			for curr_time_step in range(self.num_time_steps):

				if curr_time_step == 0:
					pass
				else:
					tf.get_variable_scope().reuse_variables()

				word_at_current_time_step = _input_[:,curr_time_step,:]

				(lstm_cell_output, curr_lstm_state) = self.multi_layer_Cell(word_at_current_time_step, curr_lstm_state)

				lstm_outputs.append(lstm_cell_output)
				lstm_states.append(curr_lstm_state)

		self.final_LSTM_state = lstm_states[-1]

		return self.final_LSTM_state, lstm_outputs



	def back_propagation_pass(self):

		# setup learning rate variable to decay
		self.learn_rate = tf.Variable(self.learning_rate, trainable=False)

		# define training operation and clip the gradients
		train_variables = tf.trainable_variables()
		gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_variables), self.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
		
		self.network_optimizer = optimizer.apply_gradients(zip(gradients, train_variables), name="train")





	def begin_training(self, training_data, validation_data, num_epochs, dropout_keep_prob, batch_size):

		self.batch_size = batch_size

		self.dropout_keep_probability = dropout_keep_prob
		self.num_epochs = num_epochs

		self.train_epoch_size = ((len(training_data) // self.batch_size) - 1) // self.num_time_steps

		training_data = np.asarray(training_data)
		validation_data = np.asarray(validation_data)

		self.train_data = training_data
		self.valid_data = validation_data

		self.run_Session()




	def run_Session(self):


		for i in range(self.num_epochs):

			# RUN TARINING PASS

			print("@ Epoch: %d Learning Rate now equals : %.3f" % (i + 1, self.sess.run(self.learn_rate)))

			# accumulated counts
			total_costs = 0.0
			total_count_iters = 0

			# initial RNN state
			state = self.sess.run([self.initial_LSTM_state], feed_dict = {self.input_index_sequence : np.zeros((self.batch_size, self.num_time_steps))})
			#state = self.initial_LSTM_state.eval()

			# run training pass
			for step, (x, y) in enumerate(ptb_reader.ptb_iterator(self.train_data, self.batch_size, self.num_time_steps)):
				
				_feed_dict_ = {self.input_index_sequence: x, self.target_scores: y, self.initial_LSTM_state: state, self.dropout_keep_prob : self.dropout_keep_probability}

				cost, state, _ = self.sess.run([self.cost, self.final_LSTM_state, self.network_optimizer], feed_dict=_feed_dict_)
				total_costs += cost
				total_count_iters += self.num_time_steps

				perplexity = np.exp(total_costs / total_count_iters)

				if step % 50 == 0:
					progress = (step / self.train_epoch_size) * 100
					print("Completed %.1f%% : Perplexity: %.3f (Cost: %.3f)" % (progress, perplexity, cost))

			print('\nCompleted a Training epoch with Net Average Cost = ', (total_costs / total_count_iters) ,'& Average Perplexity = ', perplexity ,'... \n\n ')



			# RUN VALIDATION PASS

			# accumulated counts
			total_costs = 0.0
			total_count_iters = 0

			# initial RNN state
			state = self.sess.run([self.initial_LSTM_state], feed_dict = {self.input_index_sequence : np.zeros((self.batch_size, self.num_time_steps))})
			#state = self.initial_LSTM_state.eval()
			
			for step, (x, y) in enumerate(ptb_reader.ptb_iterator(self.valid_data, self.batch_size, self.num_time_steps)):
				
				_feed_dict_ = {self.input_index_sequence: x, self.target_scores: y, self.initial_LSTM_state: state, self.dropout_keep_prob : 1.0}

				cost, state = self.sess.run([self.cost, self.final_LSTM_state], feed_dict=_feed_dict_) 	# DO NOT RUN OPTIMIZER becoz it is a Validation Pass
				total_costs += cost
				total_count_iters += self.num_time_steps

				perplexity = np.exp(total_costs / total_count_iters)

			print('\nCompleted a Validation Run with Net Average Cost = ', (total_costs / total_count_iters) ,'& Average Perplexity = ', perplexity ,'... \n\n ')

			self.save_model()

		

	def save_model(self):

		save_path = self.saver.save(self.sess, "weights/"+self.model_name+".ckpt")
		print("Model has been succesfully saved to the file: %s" % save_path)


	def test_Network(self, test_data):

		test_data = np.asarray(test_data)

		# RUN TEST PASS

		# accumulated counts
		total_costs = 0.0
		total_count_iters = 0

		# initial RNN state
		state = self.sess.run([self.initial_LSTM_state], feed_dict = {self.input_index_sequence : np.zeros((1, self.num_time_steps))})
		#state = self.initial_LSTM_state.eval()
		
		for step, (x, y) in enumerate(ptb_reader.ptb_iterator(test_data, 1, self.num_time_steps)): # Both Batch Size & Number of time Steps is 1 for Test Phase
			
			_feed_dict_ = {self.input_index_sequence: x, self.target_scores: y, self.initial_LSTM_state: state, self.dropout_keep_prob : 1.0}

			cost, state = self.sess.run([self.cost, self.final_LSTM_state], feed_dict=_feed_dict_) 	# DO NOT RUN OPTIMIZER becoz it is a Test Pass
			total_costs += cost
			total_count_iters += self.num_time_steps

			perplexity = np.exp(total_costs / total_count_iters)

		print('\nCompleted TEST PHASE ... Results -> Net Average Cost = ', (total_costs / total_count_iters) ,'& Average Perplexity = ', perplexity ,'... \n\n ')


		return (total_costs / total_count_iters), perplexity


	# def get_Output_Vector(self, test_data):

	# 	x_test = np.asarray(test_data)

	# 	[output] = self.sess.run([self.Y_], {self.X : x_test})

	# 	return output




