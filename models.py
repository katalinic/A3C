import tensorflow as tf
import numpy as np

class SimpleModel(object):
    def __init__(self, obs_size, action_size):
        self.x = x = tf.placeholder(tf.float32, [None, obs_size])
        z = tf.contrib.layers.fully_connected(x, 20, activation_fn = tf.nn.relu) #note the hardcoding here!
        logits = tf.contrib.layers.fully_connected(z, action_size, activation_fn = None)
        self.probs = tf.nn.softmax(logits)
        value = tf.contrib.layers.fully_connected(z, 1, activation_fn = None)
        self.value = value[:,0]
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    def policy_and_value(self, sess, state, mode='both'): #if threading runs on single session, can omit passing session
        if mode=='both':
            return sess.run([self.value, self.probs], feed_dict = {self.x : state})
        elif mode=='value':
            return sess.run(self.value, feed_dict = {self.x : state})
        elif mode=='prob':
            return sess.run(self.probs, feed_dict = {self.x : state})
        else:
            pr = sess.run(self.probs, feed_dict = {self.x : state})
            action = np.random.choice(self.action_size, p=pr.ravel())
            return action

class CNNModel(object):
    def __init__(self, obs_size, action_size):
        #for CNN and atari obs should be 84 84 4
        self.action_size = action_size
        self.x = x = tf.placeholder(tf.float32, [None, *obs_size])
        conv1 = tf.layers.conv2d(x, filters=16, kernel_size=[8,8], strides=(4,4), activation = tf.nn.relu) #20 20 16
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=[4,4], strides=(2,2), activation = tf.nn.relu) #9, 9, 32
        flattened = tf.reshape(conv2, [-1, 9*9*32])
        z = tf.contrib.layers.fully_connected(flattened, 256, activation_fn = tf.nn.relu)
        logits = tf.contrib.layers.fully_connected(z, action_size, activation_fn = None)
        self.probs = tf.nn.softmax(logits)
        value = tf.contrib.layers.fully_connected(z, 1, activation_fn = None)
        self.value = value[:,0]
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    def policy_and_value(self, sess, state, mode='both'): #if threading runs on single session, can omit passing session
        if mode=='both':
            return sess.run([self.value, self.probs], feed_dict = {self.x : state})
        # else: return np.argmax(sess.run([self.probs], feed_dict = {self.x : state}))
        elif mode=='value':
            return sess.run(self.value, feed_dict = {self.x : state})
        elif mode=='prob':
            return sess.run(self.probs, feed_dict = {self.x : state})
        else:
            pr = sess.run(self.probs, feed_dict = {self.x : state})
            action = np.random.choice(self.action_size, p=pr.ravel())
            return action
# cnn = CNNModel((84,84,4),4)
