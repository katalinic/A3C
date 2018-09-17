import tensorflow as tf

# from scipy.misc import imresize
import cv2

import numpy as np
import math

class TFEnv(object):
    def __init__(self, env, frame_stack=4, noop=30):
        self.env = env
        self._frame_stack = frame_stack
        self._lum_w = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)
        self._noop = noop

        self._obs = tf.get_variable(
            'obs', shape=[84, 84, 4], dtype=tf.float32, initializer=tf.zeros_initializer(),
            trainable=False, use_resource=True)
        self._reward = tf.get_variable(
            'reward', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(0),
            trainable=False, use_resource=True)
        self._done = tf.get_variable(
            'done', shape=[], dtype=tf.bool, initializer=tf.constant_initializer(True),
            trainable=False, use_resource=True)


    def _preprocess(self, x):
        x = tf.einsum('ijk,k->ij', x, self._lum_w)  # 210 160
        # x = tf.py_func(lambda z: imresize(z, (84, 84), mode='F'), [x], tf.float32)
        x = tf.py_func(lambda z: cv2.resize(z, (84, 84), interpolation=cv2.INTER_AREA),
            [x], tf.float32)
        x /= 255.0
        x = tf.expand_dims(x, -1)  # 84 84 1
        return x

    def step(self, action):
        raw_obs, reward, done = tf.py_func(
            lambda a: self.env.step(a)[:3], [action], [tf.uint8, tf.float64, tf.bool])
        reward = tf.cast(reward, tf.float32)
        raw_obs = tf.cast(raw_obs, tf.float32)
        raw_obs.set_shape([210, 160, 3])
        processed_obs = self._preprocess(raw_obs)
        stacked_obs = tf.tile(processed_obs, (1, 1, self._frame_stack))  # 84 84 4
        stacked_obs = tf.concat([self._obs[..., 1:], processed_obs], axis=2)
        return tf.group(
            self._obs.assign(stacked_obs),
            self._reward.assign(reward),
            self._done.assign(done))

    def reset(self):
        # No-op.
        if 'FIRE' in self.env.unwrapped.get_action_meanings():
            def noop_fn():
                _obs = self.env.reset()
                self.env.step(1)
                for _ in range(np.random.randint(self._noop)):
                    _obs = self.env.step(0)[0]
                return _obs
            raw_obs = tf.py_func(noop_fn, [], tf.uint8)
        else:
            raw_obs = tf.py_func(self.env.reset, [], tf.uint8)
        raw_obs = tf.cast(raw_obs, tf.float32)
        raw_obs.set_shape([210, 160, 3])
        processed_obs = self._preprocess(raw_obs)
        processed_obs = tf.tile(processed_obs, (1, 1, 4))  # 84 84 4
        with tf.control_dependencies([
            self._obs.assign(processed_obs),
            self._reward.assign(0),
            self._done.assign(False)
            ]):
            return tf.identity(processed_obs)

    @property
    def obs(self):
        return self._obs

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return self._done


# Tensorflow version
# import numpy as np
# import gym
# env = gym.make('BreakoutDeterministic-v4')
# TF_env = TFEnv(env, noop=30)
# sess = tf.Session()
#
# env_reset_op = TF_env.reset()
#
# # Breakout-specific test.
# action = tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32)
# action = action[0]
# action = tf.constant(1, tf.int32)
# env_step_op = TF_env.step(action)
#
#
# sess.run(tf.global_variables_initializer())
#
# obs = sess.run(env_reset_op)
# all_obs = [obs[..., -1]]  # Processed obs received will be in final spot.
#
# eps=0
# while eps<2:
#     obs, _, done = sess.run([TF_env.obs, env_step_op, TF_env.done])
#     all_obs.append(obs[..., -1])
#     if done:
#         eps+=1
#         sess.run(env_reset_op)
# print('here')
#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# def animate_image(image_arr):
#     fig = plt.figure()
#     all_frames = [[plt.imshow(m)] for m in image_arr]
#     ani = animation.ArtistAnimation(fig, all_frames, interval=200, blit=False, repeat=False)
#     plt.show()
#
# animate_image(all_obs)
