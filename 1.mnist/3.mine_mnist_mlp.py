# -*- coding: utf-8 -*-
import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
two fully connected layer
use dropout and relu
accuracy 0.9641
'''
def mlp(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    w = tf.Variable(tf.truncated_normal([784, 625], stddev=0.05))
    b = tf.Variable(tf.truncated_normal([625], stddev=0.05))
    y1 = tf.add(tf.matmul(x, w), b)
    y1 = tf.nn.relu(y1)
    y1 = tf.nn.dropout(y1, keep_prob)

    w2 = tf.Variable(tf.zeros([625, 10]))
    b2 = tf.Variable(tf.zeros([10]))
    y = tf.add(tf.matmul(y1, w2), b2)
    # y2 = tf.nn.relu(y2)
    # y2 = tf.nn.dropout(y2, keep_prob)
    #
    # w3 = tf.Variable(tf.zeros([400, 10]))
    # b3 = tf.Variable(tf.zeros([10]))
    # y = tf.add(tf.matmul(y2, w3), b3)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    varinit = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(varinit)

        for i in xrange(1000):
            x_info, y_info = mnist.train.next_batch(100)
            sess.run(train, feed_dict={x: x_info, y_: y_info, keep_prob: 0.5})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels,
                                            keep_prob: 1}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/1.mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=mlp, argv=[sys.argv[0]] + unparsed)