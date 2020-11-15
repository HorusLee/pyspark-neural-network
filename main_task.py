# -*- coding: utf-8 -*-

"""
@author: Zihao Li
Class: CS777 Big Data
Date: Sat Aug 15, 2020
Assignment Project
Description of Problem:
Neural Network with VCE
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import sys

from operator import add
from pyspark import SparkContext


def frequency_array(list_of_indices):
	"""build array function"""
	return_val = np.zeros(10000)
	for index in list_of_indices:
		return_val[index] += 1
	my_sum = np.sum(return_val)
	return np.divide(return_val, my_sum)


def label_feature(corpus, dictionary=None):
	"""get label_feature rdd for a given text corpus"""
	# each entry in valid_lines will be a line from the text file
	valid_lines = corpus.filter(lambda x: 'id' in x and 'url=' in x)
	key_and_text = valid_lines.map(lambda x: (
		x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))

	# we use a regular expression here
	regex = re.compile('[^a-zA-Z]')

	# remove all non letter characters
	key_and_list_of_words = key_and_text. \
		map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	key_and_list_of_words.cache()

	if dictionary is None:
		# now get the top 10,000 words
		all_words = key_and_list_of_words. \
			flatMap(lambda x: x[1]).map(lambda x: (x, 1))
		all_counts = all_words.reduceByKey(add)
		top_words = all_counts.top(10000, lambda x: x[1])

		# 10,000 is the number of words that will be in our dictionary
		ten_k = sc.parallelize(range(10000))
		dictionary = ten_k.map(lambda x: (top_words[x][0], x))

	# get a rdd ('word1', doc_id), ('word2', doc_id)
	all_words_with_doc_id = key_and_list_of_words. \
		flatMap(lambda x: ((k, x[0]) for k in x[1]))

	# join and link them, to get a set of ('word1', (dictionary_pos, doc_id))
	all_dictionary_words = dictionary.join(all_words_with_doc_id)

	# we drop the actual word itself to get a set of (doc_id, dictionary_pos)
	just_doc_and_pos = all_dictionary_words.map(lambda x: (x[1][1], x[1][0]))

	# now get a set of (doc_id, [dictionary_pos1, dictionary_pos2...]) pairs
	all_dictionary_words_in_each_doc = just_doc_and_pos.groupByKey()

	# converts the dictionary positions to a bag-of-words numpy array...
	all_docs_as_numpy_arrays = all_dictionary_words_in_each_doc. \
		map(lambda x: (1 if x[0][:2] == 'AU' else 0, frequency_array(x[1])))

	return all_docs_as_numpy_arrays, dictionary


def sigmoid(x):
	"""compute the sigmoid of x"""
	return 1 / (1 + np.exp(-x))


def theta(x):
	"""compute the theta of x"""
	return (np.sign(x) + 1) / 2


def initialize_parameters(n_x, n_h, n_y):
	"""initialize the model's parameters"""
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros((n_y, 1))

	# create parameters dictionary to store them
	parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
	return parameters


def forward_propagation(Y_X_rdd, parameters):
	"""forward propagation function to get Z1, A1, Z2, A2"""
	# retrieve each parameter from the dictionary 'parameters'
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']

	# implement forward propagation to calculate A2 (probabilities or y_hat)
	# compute Z1 = W1 * X + b1
	Y_X_Z1 = Y_X_rdd.map(lambda x: (x[0], x[1], np.dot(W1, x[1]) + b1.T))

	# compute A1 = tanh(Z1)
	Y_X_Z1_A1 = Y_X_Z1.map(lambda x: (x[0], x[1], x[2].T, np.tanh(x[2].T)))

	# compute Z2 = W2 * A1 + b2
	Y_X_Z1_A1_Z2 = Y_X_Z1_A1. \
		map(lambda x: (x[0], x[1], x[2], x[3], np.dot(W2, x[3]) + b2))

	# compute A2 = sigmoid(Z2)
	Y_X_Z1_A1_Z2_A2 = Y_X_Z1_A1_Z2. \
		map(lambda x: (x[0], x[1], x[2], x[3], x[4], sigmoid(x[4])))
	Y_X_Z1_A1_Z2_A2.cache()
	return Y_X_Z1_A1_Z2_A2


def compute_variant_cost(Y_X_Z1_A1_Z2_A2):
	"""computes the variant binary-cross-entropy cost"""
	# L = -neg_ratio * theta(m - y_hat) * y * log(y_hat)
	#     -pos_ratio * theta(y_hat - 1 + m) * (1 - y) * log(1 - y_hat)
	cost = -Y_X_Z1_A1_Z2_A2.map(lambda x: neg_ratio * theta(
		margin - x[5]) * x[0] * np.log(x[5] + 1e-9) + pos_ratio * theta(
		x[5] - 1 + margin) * (1 - x[0]) * np.log(1 - x[5] + 1e-9)).reduce(add)

	# get cost from np.array: [[cost]] -> cost
	cost = float(np.squeeze(cost))
	assert (isinstance(cost, float))
	return cost


def backward_propagation_variant(Y_X_Z1_A1_Z2_A2, parameters):
	"""implement the variant backward propagation to get dW1, db1, dW2, db2"""
	# retrieve W2 from the dictionary parameters
	W2 = parameters['W2']

	# compute dZ2 = pos_ratio * theta(A2 - 1 + m) * (1 - Y) * A2 -
	#               neg_ratio * theta(m - A2) * Y * (1 - A2)
	X_A1_dZ2 = Y_X_Z1_A1_Z2_A2.map(lambda x: (
		x[1], x[3], pos_ratio * theta(x[5] - 1 + margin) * (1 - x[0]) * x[5] -
		neg_ratio * theta(margin - x[5]) * x[0] * (1 - x[5])))

	# compute dW2 = dZ2 * A1.T
	X_A1_dZ2_dW2 = X_A1_dZ2.map(lambda x: (x[0], x[1], x[2], x[2] * x[1].T))

	# compute dZ1 = W2.T * dZ2 * (1 - A1 ^ 2)
	X_dZ2_dW2_dZ1 = X_A1_dZ2_dW2. \
		map(lambda x: (x[0], x[2], x[3], W2.T * x[2] * (1 - np.power(x[1], 2))))

	# compute dW1 = dZ1 * X
	dZ2_dW2_dZ1_dW1 = X_dZ2_dW2_dZ1. \
		map(lambda x: (x[1] / m, x[2] / m, x[3] / m, x[3] * x[0] / m))

	# get all gradients: db2, dW2, db1, dW1
	db2, dW2, db1, dW1 = dZ2_dW2_dZ1_dW1.reduce(lambda x1, x2: (
		x1[0] + x2[0], x1[1] + x2[1], x1[2] + x2[2], x1[3] + x2[3]))
	gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
	return gradients


def update_parameters(parameters, gradients, learning_rate=1.0):
	"""updates parameters using the gradient descent update rule"""
	# retrieve each parameter from the dictionary parameters
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']

	# retrieve each gradient from the dictionary gradients
	dW1 = gradients['dW1']
	db1 = gradients['db1']
	dW2 = gradients['dW2']
	db2 = gradients['db2']

	# update rule for each parameter
	W1 -= learning_rate * dW1
	b1 -= learning_rate * db1
	W2 -= learning_rate * dW2
	b2 -= learning_rate * db2

	parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
	return parameters


def mini_batch(positive_rdd, negative_rdd, num_iterations):
	"""get the positive and negative classes with index for mini-batch"""
	# mini-batch preparation
	pos_num = int(batch_size / 46)
	neg_num = pos_num * 45
	i = num_iterations % int(74 / pos_num)

	# get the new mini-batch rdd for this iteration
	new_rdd = positive_rdd. \
		filter(lambda x: i * pos_num <= x[1] < (i + 1) * pos_num). \
		map(lambda x: (x[0][0], x[0][1])).union(
			negative_rdd.filter(
				lambda x: i * neg_num <= x[1] < (i + 1) * neg_num).map(
				lambda x: (x[0][0], x[0][1])))
	return new_rdd


def neural_network(training_rdd, n_h, num_iterations=400, print_cost=False):
	"""the neural network model has to use the previous functions"""
	# initialize parameters
	learning_rate = 10.0 * n_batch
	n_x = 10000
	n_y = 1
	old_cost = 0
	parameters = initialize_parameters(n_x, n_h, n_y)

	# split the positive and negative samples in different rdd
	positive_rdd = training_rdd.filter(lambda x: x[0] == 1).zipWithIndex()
	negative_rdd = training_rdd.filter(lambda x: x[0] == 0).zipWithIndex()
	positive_rdd.cache()
	negative_rdd.cache()

	# loop for gradient descent
	for i in range(num_iterations):

		# get the new rdd for this iteration
		# new_rdd = mini_batch(positive_rdd, negative_rdd, num_iterations)

		# forward propagation
		Y_X_Z1_A1_Z2_A2 = forward_propagation(training_rdd, parameters)

		# cost function
		cost = compute_variant_cost(Y_X_Z1_A1_Z2_A2)

		# update learning rate by using "bold driver" technique
		learning_rate *= 1.05 \
			if old_cost > cost and learning_rate < (400 - i) * n_batch else .5
		old_cost = cost

		# backpropagation
		gradients = backward_propagation_variant(Y_X_Z1_A1_Z2_A2, parameters)

		# gradient descent parameter update
		parameters = update_parameters(parameters, gradients, learning_rate)

		# make predictions on test data
		accuracy, f1_measure, _ = predict(parameters)

		# print the cost every 10 iterations
		if print_cost:
			print("Iteration: {}, Cost: {}, Accuracy: {}, F1: {}".
			      format(i, cost, accuracy, f1_measure))

		# stop when cost is not changing so much
		if accuracy > 0.99:
			print("Stopped at iteration {}".format(i))
			probability_plot(parameters)
			break
	return parameters


def predict(parameters):
	"""using the learned parameters, predicts a class for each example"""
	Y_X_Z1_A1_Z2_A2 = forward_propagation(test, parameters)
	Y_prediction = Y_X_Z1_A1_Z2_A2. \
		map(lambda x: (x[0], 1 if x[5] > 0.5 else 0))

	# compute tp, tn, fp, fn from y and predictions
	tp_tn_fp_fn = Y_prediction. \
		map(lambda x: np.array([x[0] * x[1], (1 - x[0]) * (1 - x[1]), (
			1 - x[0]) * x[1], x[0] * (1 - x[1])])).reduce(add)
	tp, tn, fp, fn = tp_tn_fp_fn
	Y_A2 = Y_X_Z1_A1_Z2_A2.map(lambda x: (x[0], float(np.squeeze(x[5]))))

	# calculate the accuracy and f1 measure
	accuracy = (tp + tn) / n_test
	precision = tp / (tp + fp + 1e-9)
	recall = tp / (tp + fn + 1e-9)
	f1_measure = 2 * precision * recall / (precision + recall + 1e-9)
	return accuracy, f1_measure, Y_A2


def probability_plot(parameters):
	"""plot the probabilities for all the predictions"""
	_, _, Y_A2 = predict(parameters)
	label = np.array(Y_A2.map(lambda x: x[0]).collect())
	probability = np.array(Y_A2.map(lambda x: x[1]).collect())
	point = range(n_test)

	# plot the probabilities
	plt.scatter(point, probability, c=1 - label, s=1)
	plt.plot(point, np.ones(len(point)) / 2, color='k')
	plt.ylabel('probability')
	plt.savefig('probability.png')
	plt.show()


if __name__ == '__main__':
	# file path:
	sc = SparkContext()
	small_data = "/Users/Horus/Desktop/Horus/Computer Science/" \
	             "CS777-Big Data Analytics/Project/SmallTrainingData.txt"
	large_training_data = sys.argv[1]
	large_test_data = sys.argv[2]
	training_data = small_data
	test_data = small_data

	# get the training and test dataset
	training_corpus = sc.textFile(training_data, 1)
	training, dic = label_feature(training_corpus)
	training.cache()
	dic.cache()
	test_corpus = sc.textFile(test_data, 1)
	test, _ = label_feature(test_corpus, dic)
	test.cache()

	# initialize hyper-parameters
	batch_size = 512  # 46 <= batch size <= 3442
	hidden_unit = 1  # only 1 hidden unit can get 99% accuracy with 0.78 F1
	iteration = 300
	margin = 0.9
	m = training.count()
	n_batch = 1
	n_positive = training.map(lambda x: x[0]).reduce(add)
	n_test = test.count()
	pos_ratio = n_positive / m
	neg_ratio = 1 - pos_ratio
	parameter = neural_network(training, hidden_unit, iteration, True)
