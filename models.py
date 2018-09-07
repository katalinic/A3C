import tensorflow as tf

class CNNModel(object):
    def __init__(self, obs_size, action_size):
        # Atari input is 84 84 4
        self.action_size = action_size
        self.x = x = tf.placeholder(tf.float32, [None, *obs_size])
        x = tf.layers.conv2d(x, filters=16, kernel_size=[8, 8],
                             strides=(4, 4), activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=32, kernel_size=[4, 4],
                             strides=(2, 2), activation=tf.nn.relu)
        x = tf.reshape(x, [-1, 9 * 9 * 32])
        x = tf.contrib.layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
        self.logits = tf.contrib.layers.fully_connected(x, action_size, activation_fn=None)
        self.probs = tf.nn.softmax(self.logits)
        action = tf.multinomial(self.logits, num_samples=1, output_dtype=tf.int32)
        self.action = tf.squeeze(action)
        value = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
        self.value = value[:, 0]
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=tf.get_variable_scope().name)

    def policy_and_value(self, sess, state, mode='both'):
        if mode == 'both':
            return sess.run([self.value, self.action], feed_dict={self.x: state})
        elif mode == 'value':
            return sess.run(self.value, feed_dict={self.x: state})
        elif mode == 'prob':
            return sess.run(self.probs, feed_dict={self.x: state})
        else:
            action = sess.run(self.action, feed_dict={self.x: state})
            return action
