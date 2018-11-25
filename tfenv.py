import tensorflow as tf
import cv2
import numpy as np

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
        stacked_obs = tf.cond(done, lambda: self.reset(), lambda: stacked_obs)
        with tf.control_dependencies([stacked_obs]):
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
