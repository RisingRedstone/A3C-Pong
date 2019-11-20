#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:55:32 2019

@author: risingredstone
"""

import numpy as np
import gym
import multiprocessing as mp
import random
import cv2

class Stack:
    
    def __init__(self, shape, maxFrames):
        self.shape = shape
        self.maxFrames = maxFrames
        self.frames = []
    
    def add(self, value):
        assert np.shape(value) == self.shape
        self.frames.append(value)
        if len(self.frames) > self.maxFrames:
            self.frames.pop(0)
    
    def reset(self):
        self.frames = []
    
    def read(self):
        assert len(self.frames) == self.maxFrames
        return np.moveaxis(self.frames, 0, -1)

class Environment:
    
    def __init__(self, env_name, render, pipe):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render = render
        self.pipe  = pipe
        
    def run(self):
        while True:
            if self.render:
                self.env.render()
            cmd, data = self.pipe.recv()
            if cmd == 'step':
                self.pipe.send(self.env.step(data))
            elif cmd == 'reset':
                self.pipe.send(self.env.reset())

class EnvironmentProcess:
    
    def __init__(self, env_name, render, shape, maxFrames):
        self.stack = Stack(shape, maxFrames)
        self.shape = shape
        self.maxFrames = maxFrames
        self.env_name = env_name
        self.render = render
        p1, p2 = mp.Pipe()
        self.pipe = p1
        Env = Environment(env_name, render, p2)
        self.process = mp.Process(target = Env.run)
        self.process.start()
    
    def preprocess(self, obs):
        '''Processes the a frame'''
        obs = np.mean(obs, axis=2) / 255.0
        obs = obs[34:194]
#        obs = obs[::2, ::2]
#        obs = np.pad(obs, pad_width=2, mode='constant')
#        obs[obs <= 0.4] = 0
#        obs[obs > 0.4] = 1
        obs = cv2.resize(obs, (84, 84), interpolation = cv2.INTER_NEAREST)
        return obs
    
    def reset(self):
        self.stack.reset()
        
        self.pipe.send(('reset', None))
        obs = self.pipe.recv()
        obs = self.preprocess(obs)
        self.stack.add(obs)
        
        #Random Number of noops
        for _ in range(random.randint(0, 30)):
            self.pipe.send(('step', 0))
            self.pipe.recv()
        
        for _ in range((self.maxFrames - 1)):
            self.pipe.send(('step', 0))
            obs, reward, done, info = self.pipe.recv()
            obs = self.preprocess(obs)
            self.stack.add(obs)
            if done:
                raise Exception("Environment is already done during initial frame stack")
        
        return self.stack.read()
    
    def step(self, action):
        rewardSum = 0
        
        #frame Skipping
        #You repeat the same action a number of times
        for _ in range(4):
            self.pipe.send(('step', action))
            obs, reward, done, info = self.pipe.recv()
            rewardSum += reward
            obs = self.preprocess(obs)
            self.stack.add(obs)
            if done:
                break
        return (self.stack.read(), rewardSum, done, info)
    
    def close(self):
        self.process.terminate()
    
    
    
