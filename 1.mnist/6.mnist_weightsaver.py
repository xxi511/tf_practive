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

    w = tf.Variable(tf.truncated_normal([784, 625], stddev=0.05), name='w1')
    b = tf.Variable(tf.truncated_normal([625], stddev=0.05), name='b1')
    y1 = tf.add(tf.matmul(x, w), b)
    y1 = tf.nn.relu(y1)
    y1 = tf.nn.dropout(y1, keep_prob)

    w2 = tf.Variable(tf.zeros([625, 10]), name='w2')
    b2 = tf.Variable(tf.zeros([10]), name='b2')
    y = tf.add(tf.matmul(y1, w2), b2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    '''
    tf.train.Saver.__init__(var_list=None,
                            reshape=False,
                            sharded=False,
                            max_to_keep=5,   # maximum number of recent checkpoint files to keep
                            keep_checkpoint_every_n_hours=10000.0,  # keep one checkpoint file for every N hours of training
                            name=None,
                            restore_sequentially=False,
                            saver_def=None,
                            builder=None,
                            defer_build=False,
                            allow_empty=False,
                            write_version=2,
                            pad_step_number=False)
    '''

    saver = tf.train.Saver(var_list={'w1': w, 'b1': b, 'w2': w2, 'b2': b2})

    varinit = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(varinit)

        for i in xrange(1000):
            x_info, y_info = mnist.train.next_batch(100)
            sess.run(train, feed_dict={x: x_info, y_: y_info, keep_prob: 0.5})

        saver.save(sess, './mlpmodel.ckpt', 1000)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels,
                                            keep_prob: 1}))

def weight_restore(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    w = tf.Variable(tf.truncated_normal([784, 625], stddev=0.05), name='w1')
    b = tf.Variable(tf.truncated_normal([625], stddev=0.05), name='b1')
    y1 = tf.add(tf.matmul(x, w), b)
    y1 = tf.nn.relu(y1)
    y1 = tf.nn.dropout(y1, keep_prob)

    w2 = tf.Variable(tf.zeros([625, 10]), name='w2')
    b2 = tf.Variable(tf.zeros([10]), name='b2')
    y = tf.add(tf.matmul(y1, w2), b2)

    # restore failed
    # w_extra = tf.Variable(tf.zeros[34], name='aaa')

    saver = tf.train.Saver()

    varinit = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(varinit)

        ckpt = tf.train.get_checkpoint_state('./')

        if ckpt and ckpt.model_checkpoint_path:
            print 'got ckpt'

            saver.restore(sess, ckpt.model_checkpoint_path)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                y_: mnist.test.labels,
                                                keep_prob: 1}))
        else:
            print 'no ckpt exist'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/1.mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    #tf.app.run(main=mlp, argv=[sys.argv[0]] + unparsed)

    tf.app.run(main=weight_restore, argv=[sys.argv[0]] + unparsed)