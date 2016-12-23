# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    # x is input images, 2D matrix[batch, flatten size(28*28)]
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))  # weight
    b = tf.Variable(tf.zeros([10]))  # biases
    # W = tf.Variable(tf.truncated_normal([784, 10]))
    # b = tf.Variable(tf.truncated_normal([10]))
    y = tf.matmul(x, W) + b  # predict answer

    # Define loss and optimizer
    # real answer, 2D matrix[batch, 10] (one hot encode)
    y_ = tf.placeholder(tf.float32, [None, 10])

    '''
    tf.nn.softmax_cross_entropy_with_logits(logits, labels, dim=-1, name=None)
    Computes softmax cross entropy between logits(predict) and labels.

    tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
    calculate mean across tensor
    '''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

    '''
    use GradientDescent as weight optimizer, learning rate is 0.5
    goal is minimize cross_entropy
    '''
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    # initial all Variables
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):  # 1000 for loop
        # batch size is 100 for each loop
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
