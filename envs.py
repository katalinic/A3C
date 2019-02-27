import tensorflow as tf

import gym

from pyprocess import PyProcess
from preprocessing import atari_preprocess

OBS_SPACE = [84, 84, 4]
SPECS = {
    'reset': ([tf.float32, tf.float32, tf.bool], [OBS_SPACE, [], []]),
    'step': ([tf.float32, tf.float32, tf.bool], [OBS_SPACE, [], []])}


def create_env(env_name):
    env_ = gym.make(env_name)
    action_space = env_.action_space.n
    env_ = atari_preprocess(env_)
    env_.specs = SPECS
    env_ = PyProcess(env_)
    env_.start()
    return env_, action_space
