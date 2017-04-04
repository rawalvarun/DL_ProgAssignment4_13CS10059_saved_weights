

# parses the dataset
import ptb_reader


import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import sys
import math
import os
import configparser as ConfigParser
import random
import time

import VR_tf_PTB_LSTM as LSTM_NN


import sys

'''
This is the main logic for serializing and deserializing dictionaries
of hyperparameters (for use in checkpoint restoration and sampling)
'''
import os
import pickle


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")



def read_config_file():
	'''
	Reads in config file, returns dictionary of network params
	'''

	config = ConfigParser.ConfigParser()
	config.read(FLAGS.config_file)

	dic = {}

	__model_init__ = "model_initialization_hyperparameters"
	__trainer__ = "model_training_hyperparameters"
	__general__ = "general_interface_options"

	dic["batch_size"] = config.getint(__model_init__, "batch_size")
	dic["num_steps"] = config.getint(__model_init__, "num_steps")
	dic["num_layers"] = config.getint(__model_init__, "num_layers")
	dic["hidden_size"] = config.getint(__model_init__, "hidden_size")
	dic["vocab_size"] = config.getint(__model_init__, "vocab_size")
	dic["max_grad_norm"] = config.getint(__model_init__, "max_grad_norm")
	dic["init_scale"] = config.getfloat(__model_init__, "init_scale")

	dic["keep_prob"] = config.getfloat(__trainer__, "keep_prob")
	dic["learning_rate"] = config.getfloat(__trainer__, "learning_rate")
	dic["lr_decay"] = config.getfloat(__trainer__, "lr_decay")
	dic["lr_decay_epoch_offset"] = config.getint(__trainer__, "lr_decay_epoch_offset")

	dic["verbose_mode"] = config.getint(__general__, "verbose_mode")
	dic["enable_training"] = config.getint(__general__, "enable_training")

	return dic



hyper_params = read_config_file()


flags.DEFINE_integer("batch_size", hyper_params["batch_size"], "Size of a trainable Batch")
flags.DEFINE_integer("num_steps", hyper_params["num_steps"], "Number of unrolled time steps")
flags.DEFINE_integer("num_layers", hyper_params["num_layers"], "number of hidden LSTM layers")
flags.DEFINE_integer("hidden_size", hyper_params["hidden_size"], "number of blocks in an LSTM cell")
flags.DEFINE_integer("vocab_size", hyper_params["vocab_size"], "Size of Vocabulary collected across Data")
flags.DEFINE_integer("max_grad_norm", hyper_params["max_grad_norm"], "maximum gradient for clipping")
flags.DEFINE_float("init_scale", hyper_params["init_scale"], "scale between -0.1 and 0.1 for all random initialization")


flags.DEFINE_float("learning_rate", hyper_params["learning_rate"], "Learning rate of for adam [0.0002]")
flags.DEFINE_float("keep_prob", hyper_params["keep_prob"], "Dropout Keep Probability to avoid Overfitting while Training Phase")
flags.DEFINE_float("lr_decay", hyper_params["lr_decay"], "Decay rate of the learning rate")
flags.DEFINE_integer("lr_decay_epoch_offset", hyper_params["lr_decay_epoch_offset"] , "don't decay until after the Nth epoch")


# Save the system's standard output (terminal) file I/O
save_stdout = sys.stdout

if hyper_params['verbose_mode']==0:
	# verbose mode is disabled 

	# So, silence off the print statements
	sys.stdout = open('trash', 'w')
else:
	pass



# load dataset
train_data, valid_data, test_data, _ = ptb_reader.ptb_raw_data()



PTB_Learner_LSTM_Model = LSTM_NN.Neural_Network(model_name = 'PTB_Learner', num_time_steps = hyper_params['num_steps'], vocab_size = hyper_params['vocab_size'], hidden_layer_size = hyper_params['hidden_size'], num_hidden_layers = hyper_params['num_layers'], init_scale_half_range = hyper_params['init_scale'])
PTB_Learner_LSTM_Model.init_tensorflow_members(learning_rate = hyper_params['learning_rate'], maximum_gradient_norm = hyper_params['max_grad_norm'])


_num_epochs_ = 1


if hyper_params['enable_training'] == 1:
	PTB_Learner_LSTM_Model.begin_training(training_data = train_data, validation_data = valid_data, num_epochs = _num_epochs_, dropout_keep_prob = hyper_params['keep_prob'], batch_size = hyper_params['batch_size'])
else:
	pass


_return_result_ = PTB_Learner_LSTM_Model.test_Network(test_data)


print('Test phase returned following result : ', _return_result_ )


# Restore back the system's standard output (terminal) file I/O
sys.stdout = save_stdout

print(_return_result_[1]) # Just print the perplexity score (As Demanded in the Assignment)
