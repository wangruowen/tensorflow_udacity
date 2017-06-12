#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

start_learning_rate = 0.1
batch_size = 512
beta = 0.01 # This is a good beta value to start with.
num_steps = 20001
dropout = 0.5 # Dropout, probability to keep units
layer_node_sizes = [1024, 512, num_labels]

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape=shape))

def bias_variable(shape):
  return tf.Variable(tf.zeros(shape=shape))

weights = [
  weight_variable([image_size * image_size, layer_node_sizes[0]]),
  weight_variable([layer_node_sizes[0], layer_node_sizes[1]]),
  weight_variable([layer_node_sizes[1], layer_node_sizes[2]]),
  # weight_variable([layer_node_sizes[2], layer_node_sizes[3]])
]

biases = [
  bias_variable([layer_node_sizes[0]]),
  bias_variable([layer_node_sizes[1]]),
  bias_variable([layer_node_sizes[2]]),
  # bias_variable([layer_node_sizes[3]])
]

# dropout (keep probability), for training, we put 0.75, for validation/test, we put 1.0 for no-dropout
keep_prob = tf.placeholder(tf.float32)

def multi_layer_model(input_data):
  each_layer = input_data
  for i in range(2):
    each_layer = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(each_layer, weights[i]), biases[i])), keep_prob)
  # Apply dropout
#   each_layer = tf.nn.dropout(each_layer, keep_prob)
  # Add output layer
  return tf.add(tf.matmul(each_layer, weights[2]), biases[2])

# graph = tf.Graph()
# with graph.as_default():

# Input data. For the training data, we use a placeholder that will be fed
# at run time with a training minibatch.
tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
tf_valid_dataset = tf.constant(valid_dataset)
tf_test_dataset = tf.constant(test_dataset)

train_model = multi_layer_model(tf_train_dataset)
valid_model = multi_layer_model(tf_valid_dataset)
test_model = multi_layer_model(tf_test_dataset)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=train_model))

# Add L2 Regularization for each weight
for i in range(len(weights)):
  loss += beta * tf.nn.l2_loss(weights[i])

# Optimizer.
# gdo = tf.train.GradientDescentOptimizer
# optimizer = gdo(start_learning_rate).minimize(loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 500, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(train_model)
valid_prediction = tf.nn.softmax(valid_model)
test_prediction = tf.nn.softmax(test_model)

# show_graph(tf.get_default_graph().as_graph_def())

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as session:
  print("Initialized")
  session.run(init)

  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
#     print("Train using dataset range: " + str(offset) + "-" + str(offset + batch_size))
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {
        tf_train_dataset: batch_data,
        tf_train_labels: batch_labels,
        keep_prob: dropout
    }
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(feed_dict={keep_prob: 1.0}), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels))

