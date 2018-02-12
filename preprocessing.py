import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy.misc import imresize
# from skimage.color import rgb2xyz

'''Ignoring flickering issues (i.e. max across two steps)
Reward clipping?
'''

env = gym.make('BreakoutDeterministic-v4') #FRAME SKIP IMPLEMENTED BY DEFAULT WITH ABOVE!

class EnvWrapper():
    def __init__(self, env, pre_process = False, frame_stack = 1):
        self.env = env
        self.frame_stack = frame_stack
        self.pre_process = pre_process

    @staticmethod
    def _preprocess(M):
        y = 0.299*M[:,:,0]+0.587*M[:,:,1]+0.114*M[:,:,2]
        rescaled = imresize(y,(84,84))
        norm = rescaled/255.0
        return norm

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        #clip reward
        reward = np.clip(reward, 0, 1)
        if self.pre_process:
            obs = self._preprocess(obs)

        if self.frame_stack > 1:
            #append obs to top of array and remove first
            self.frames = np.concatenate((self.frames[:,:,1:],obs[...,np.newaxis]),axis=2)
            return self.frames, reward, done, info
        else:
            return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.pre_process:
            obs = self._preprocess(obs)
        #since upon reset there are no previous frames, we simply stack the 4
        if self.frame_stack > 1:
            self.frames = np.stack([obs for _ in range(self.frame_stack)],axis=2)
            return self.frames
        else:
            return obs

    def render(self):
        self.env.render()

# envw = EnvWrapper(env, True, 4)
# obs = env.reset()
# t = envw._preprocess(obs)
# stacked = np.stack([t for _ in range(4)],axis=2)
#
# print (np.concatenate((stacked,t[...,np.newaxis]),axis=2).shape)
#

# env = EnvWrapper(env, True, 4)
# a = env.reset()
# print (a.shape)
# for i in range(5):
#     action = 1 if i==0 else 2
#     f, _, _, _ = env.step(action)
#     print (f.shape)
