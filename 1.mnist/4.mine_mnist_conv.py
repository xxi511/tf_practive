# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
2 conv + 2 fc

'''

def weight_init(shape):
    # shape = [h, w, input_chanel, num_kernel]
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_init(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, w):
    # strides = [batch, h, w, chanel]
    # the filter won't cross batch or chanel
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # ksize: kernel size
    # [batch, h, w, chanel], create a filter just cover h and w
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

def npreshape(x):
    b = x.reshape((-1, 7*7*64))
    return b

def npreshape_grad(op, grad):
    x = op.inputs[0]
    k = tf.reshape(grad, [-1, 7, 7, 64])
    return k

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    w_conv1 = weight_init([5, 5, 1, 32])
    b_conv1 = bias_init([32])
    res_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    res_pool1 = max_pool_2x2(res_conv1)

    w_conv2 = weight_init([5, 5, 32, 64])
    b_conv2 = bias_init([64])
    res_conv2 = tf.nn.relu(conv2d(res_pool1, w_conv2) + b_conv2)
    res_pool2 = max_pool_2x2(res_conv2)
    flat_pool2 = tf.reshape(res_pool2, [-1, 7*7*64])
    # comment Line73 and uncomment Line75 to test py_func
    # flat_pool2 = py_func(npreshape, [res_pool2], [tf.float32], grad=npreshape_grad)[0]

    w_fc3 = weight_init([7*7*64, 1024])
    b_fc3 = bias_init([1024])
    res_fc3 = tf.nn.relu(tf.matmul(flat_pool2, w_fc3) + b_fc3)
    res_fc3 = tf.nn.dropout(res_fc3, keep_prob)

    w_fc4 = weight_init([1024, 10])
    b_fc4 = bias_init([10])
    y = tf.matmul(res_fc3, w_fc4) + b_fc4

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    var_init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(var_init)

        for i in xrange(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            sess.run(train, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/1.mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
