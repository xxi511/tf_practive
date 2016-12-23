# -*- coding: utf-8 -*-
import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
2 conv + 2 fc
accuracy 0.9908
'''


def weight_init(shape, name):
    # shape = [h, w, input_chanel, num_kernel]
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init, name=name)


def bias_init(shape, name):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name=name)


def conv2d(x, w, name):
    # strides = [batch, h, w, chanel]
    # the filter won't cross batch or chanel
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name):
    # ksize: kernel size
    # [batch, h, w, chanel], create a filter just cover h and w
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    '''
    use tf.name_scope to separate each block
    remember assign 'name' so that can find it on tensor board easily
    or it just show something like 'placeholder'
    '''
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        '''
        'w_conv1' is name, check weight_init
        use tf.summary.histogram to record histogram
        '''
        w_conv1 = weight_init([5, 5, 1, 32], 'w_conv1')
        tf.summary.histogram('conv1/w', w_conv1)

        b_conv1 = bias_init([32], 'b_conv1')
        tf.summary.histogram('conv1/b', b_conv1)

        res_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, 'conv1') + b_conv1)
        res_pool1 = max_pool_2x2(res_conv1, 'maxpool1')
        tf.summary.histogram('conv1/res', res_pool1)

    with tf.name_scope('conv2'):
        w_conv2 = weight_init([5, 5, 32, 64], 'w_conv2')
        tf.summary.histogram('conv2/w', w_conv2)

        b_conv2 = bias_init([64], 'b_conv2')
        tf.summary.histogram('conv2/b', b_conv2)

        res_conv2 = tf.nn.relu(conv2d(res_pool1, w_conv2, 'conv2') + b_conv2)
        res_pool2 = max_pool_2x2(res_conv2, 'maxpool2')
        tf.summary.histogram('conv2/res', res_pool2)

        flat_pool2 = tf.reshape(res_pool2, [-1, 7 * 7 * 64])

    with tf.name_scope('fc3'):
        w_fc3 = weight_init([7 * 7 * 64, 1024], 'w_fc3')
        tf.summary.histogram('fc3/w', w_fc3)

        b_fc3 = bias_init([1024], 'b_fc3')
        tf.summary.histogram('fc3/b', b_fc3)

        res_fc3 = tf.nn.relu(tf.matmul(flat_pool2, w_fc3, name='fc3') + b_fc3)
        res_fc3 = tf.nn.dropout(res_fc3, keep_prob)
        tf.summary.histogram('fc3/res', res_fc3)

    with tf.name_scope('fc4'):
        w_fc4 = weight_init([1024, 10], 'w_fc4')
        tf.summary.histogram('fc4/w', w_fc4)

        b_fc4 = bias_init([10], 'b_fc4')
        tf.summary.histogram('fc4/b', b_fc4)

        y = tf.matmul(res_fc3, w_fc4, name='fc4') + b_fc4

    with tf.name_scope('loss'):
        '''
        use tf.summary.scalar to draw line
        '''
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    '''
    at least one 'summary' (summary.scalar or summary.histogram)
    or the error occured
    '''
    merged = tf.summary.merge_all()
    var_init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(var_init)
        # save the log
        writer = tf.summary.FileWriter('logs/', sess.graph)

        for i in xrange(10000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                print i
                result = sess.run(merged, feed_dict={x: batch[0],
                                                     y_: batch[1],
                                                     keep_prob: 1.0})
                writer.add_summary(result, i)
            sess.run(train, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
