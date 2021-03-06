"""Selected subset of preprocessing steps applied to Atari.

Main step omitted is max over two consecutive frames. Action repeat
assumed via choice of gym environments.
"""

from collections import deque

import numpy as np

from gym import spaces
from gym.core import Wrapper, ObservationWrapper, RewardWrapper

import cv2


class ObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0., high=1., shape=(84, 84), dtype=np.float32)

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32)
        obs /= 255.
        return obs


class RewardFloatWrapper(RewardWrapper):
    def reward(self, reward):
        return np.float32(reward)


class NoopOnResetWrapper(Wrapper):
    def __init__(self, env, noop=30):
        super().__init__(env)
        self._noop = noop

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        if self._noop > 0:
            if 'FIRE' in self.env.unwrapped.get_action_meanings():
                obs = self.env.step(1)[0]
            for _ in range(np.random.randint(self._noop)):
                obs = self.env.step(0)[0]
        return obs


class FrameStackWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._frame_stack = deque(maxlen=4)

    def _convert_to_obs(self):
        stacked = np.array(self._frame_stack)
        stacked = np.moveaxis(stacked, 0, -1)
        return stacked

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frame_stack.append(obs)
        return self._convert_to_obs(), reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._frame_stack.append(obs)
        for _ in range(3):
            obs = self.env.step(0)[0]
            self._frame_stack.append(obs)
        return self._convert_to_obs()


class ResetOnDoneWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            self.env.reset()
        return obs, reward, done

    def reset(self):
        obs = self.env.reset()
        return obs


class EnvOutputOnResetWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        return obs, np.float32(0), False


def atari_preprocess(env, noop=30 // 4):
    env = NoopOnResetWrapper(env, noop=noop)
    env = ObsWrapper(env)
    env = RewardFloatWrapper(env)
    env = FrameStackWrapper(env)
    env = ResetOnDoneWrapper(env)
    env = EnvOutputOnResetWrapper(env)
    return env
