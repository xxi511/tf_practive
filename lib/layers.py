# -*- coding: utf-8 -*-
import tensorflow as tf


def weight_init(shape,
                truncate_init=True,
                mean=0,
                stddev=0.1,
                dtype=tf.float32,
                name=None):
    if truncate_init:
        w_init = tf.truncated_normal(shape=shape,
                                     mean=mean,
                                     stddev=stddev,
                                     dtype=dtype)
    else:
        w_init = tf.zeros(shape, dtype)

    return tf.Variable(w_init, name=name)


def bias_init(shape,
              truncate_init=False,
              mean=0,
              stddev=0.1,
              dtype=tf.float32,
              name=None
              ):
    if truncate_init:
        b_init = tf.truncated_normal(shape=shape,
                                     mean=mean,
                                     stddev=stddev,
                                     dtype=dtype)
    else:
        b_init = tf.constant(0.1, shape=shape)

    return tf.Variable(b_init, name=name)


def conv2d(t_input, nb_filter, ksize, name, stride=None,
           padding='SAME', activation=None, vis=False):
    '''

    :param t_input: input tensor
    :param nb_filter: number of filter
    :param ksize: kernel size, type is int
    :param name : name
    :param stride: [batch, h, w, chanel]
    :param padding: 'SAME' or 'VALID'
    :param activation: activation func
    :param vis: if summary histogram
    :return: output tensor
    '''
    if stride is None:
        stride = [1, 1, 1, 1]

    if padding not in {'SAME', 'VALID'}:
        raise Exception('padding should be SAME or VALID'
                        'now is {}'.format(padding))

    input_shape = t_input.get_shape()
    if input_shape.ndims != 4:
        raise Exception('input tensor dimension 4 only'
                        'now is {}'.format(input_shape.ndims))

    # filter size is [filter_height, filter_width, in_channels, out_channels]
    filter_size = [ksize, ksize, input_shape[-1].value, nb_filter]
    fileter = weight_init(filter_size, name=name + '/w')
    b = bias_init([nb_filter], name=name + '/b')

    res_conv2d = tf.nn.conv2d(t_input, fileter, stride, padding)
    res_conv2d = tf.nn.bias_add(res_conv2d, b, name=name)

    if vis:
        print '{} , filter[h,w,in,out] is {}'.format(name, filter_size)

    if activation is None:
        return res_conv2d
    else:
        return activation(res_conv2d)


def fully_connected(t_input, output_dim, name,
                    activation=None, vis=False):

    input_shape = t_input.get_shape()
    if input_shape.ndims != 2:
        raise Exception('input tensor dimension 2 only'
                        'now is {}'.format(input_shape.ndims))


    weight_shape = [input_shape[-1].value, output_dim]
    bias_shape = [output_dim]
    w = weight_init(weight_shape, name=name + '/w')
    b = bias_init(bias_shape, name=name + '/b')

    res = tf.nn.bias_add(tf.matmul(t_input, w), b, name=name)

    if vis:
        print '{} , weight[in,out] is {}'.format(name, weight_shape)

    if activation is None:
        return res
    else:
        return activation(res)
