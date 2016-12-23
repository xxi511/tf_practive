# -*- coding: utf-8 -*-
import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
just one fully connected layer
accuracy 0.919
'''

def mlp(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # accuracy is 0.91
    w = tf.Variable(tf.truncated_normal([784, 10], stddev=0.05))
    b = tf.Variable(tf.truncated_normal([10], stddev=0.05))
    # accuracy is 0.88, WT
    # w = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    y = tf.add(tf.matmul(x, w), b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    varinit = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(varinit)

        for i in xrange(1000):
            x_info, y_info = mnist.train.next_batch(100)
            if i % 100 == 0:
                print 'b is {}'.format(b.eval())
            sess.run(train, feed_dict={x: x_info, y_: y_info})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/1.mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=mlp, argv=[sys.argv[0]] + unparsed)