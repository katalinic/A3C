from collections import namedtuple

import tensorflow as tf

nest = tf.contrib.framework.nest

Policy = namedtuple('Policy', 'logits action values')


# def encoder(x):
#     initializer = tf.contrib.layers.variance_scaling_initializer(
#         factor=1.0, mode='FAN_IN', uniform=False)
#     with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
#         x = tf.expand_dims(x, 0)
#         x = tf.layers.conv2d(
#             x, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu,
#             kernel_initializer=initializer)
#         x = tf.layers.conv2d(
#             x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu,
#             kernel_initializer=initializer)
#         x = tf.layers.conv2d(
#             x, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu,
#             kernel_initializer=initializer)
#         x = tf.reshape(x, [-1, 7 * 7 * 64])
#         x = tf.layers.dense(
#             x, 512, activation=tf.nn.relu, kernel_initializer=initializer)
#         return x


def encoder(x):
    initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=1.0, mode='FAN_IN', uniform=False)
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        x = tf.expand_dims(x, 0)
        x = tf.layers.conv2d(
            x, filters=16, kernel_size=8, strides=4, activation=tf.nn.relu,
            kernel_initializer=initializer)
        x = tf.layers.conv2d(
            x, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu,
            kernel_initializer=initializer)
        x = tf.reshape(x, [-1, 9 * 9 * 32])
        x = tf.layers.dense(
            x, 256, activation=tf.nn.relu, kernel_initializer=initializer)
        return x


def head(x, action_space):
    """Policy and value networks."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=1.0, mode='FAN_IN', uniform=False)
    with tf.variable_scope('Policy', reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(x, action_space, activation=None,
                                 kernel_initializer=initializer)
        action = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
    with tf.variable_scope('Value', reuse=tf.AUTO_REUSE):
        values = tf.layers.dense(x, 1, activation=None,
                                 kernel_initializer=initializer)
    logits, action, values = nest.map_structure(
        lambda t: tf.squeeze(t), (logits, action, values))
    return Policy(logits, action, values)
