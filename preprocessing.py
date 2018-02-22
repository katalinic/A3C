import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy.misc import imresize
from gym import Wrapper
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class EnvWrapper(Wrapper):
    def __init__(self, env, pre_process = False, frame_stack = 1, noop_max = 30):
        #added for monitoring
        super(EnvWrapper, self).__init__(env)
        self.env = env
        self.frame_stack = frame_stack
        self.pre_process = pre_process
        self.noop_max = noop_max

    @staticmethod
    def _preprocess(M):
        y = 0.299*M[:,:,0]+0.587*M[:,:,1]+0.114*M[:,:,2]
        #approximate
        # y = 0.3*M[:,:,0]+0.6*M[:,:,1]+0.1*M[:,:,2]
        rescaled = imresize(y,(84,84))
        norm = rescaled/255.0
        return norm

    def step(self, action, inference=False):
        obs, reward, done, info = self.env.step(action)
        #clip reward
        if not inference: reward = np.clip(reward, -1, 1)
        if self.pre_process:
            obs = self._preprocess(obs)

        if self.frame_stack > 1:
            #append obs to top of array and remove first
            self.frames = np.concatenate((self.frames[:,:,1:],obs[...,np.newaxis]),axis=2)
            return self.frames, reward, done, info
        else:
            return obs, reward, done, info

    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1) #fire ball - breakout specific
        if self.pre_process:
            obs = self._preprocess(obs)
        #since upon reset there are no previous frames, we simply stack the 4
        if self.frame_stack > 1:
            self.frames = np.stack([obs for _ in range(self.frame_stack)],axis=2)
        #     return self.frames
        # else:
        #     return obs

        #add no-op action section
        noop = np.random.randint(math.ceil(self.noop_max//4))
        for _ in range(noop):
            if self.frame_stack > 1:
                self.frames, _, _, _ = self.step(0) #noop assumed 0
            else:
                obs, _, _, _ = self.step(0)

        if self.frame_stack > 1:
            return self.frames
        else:
            return obs

    # def render(self):
    #     return self.env.render()
