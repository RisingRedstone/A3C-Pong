#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:44:14 2019

@author: risingredstone
"""

from environment import *
from model import *
import numpy as np

globalStep = 0

class Worker:
    
    def __init__(self, sess, Env, model, TotalSteps, interval, name):
        self.sess = sess
        self.Env = Env
        self.model = model
        self.TotalSteps = TotalSteps
        self.name = name
        self.num_steps = 0
        self.interval = interval
        
    def run(self):
        global globalStep
        
        Rewards = []
        Values = []
        Actions = []
        States = []
        totReward = 0
        
        epi = 0
        
        while globalStep < self.TotalSteps:
            
            print(self.name + ":\tEpisode Number : {}\tThis is the Reward : {}".format(epi, totReward))
            epi += 1
            totReward = 0
            done = False
            
            state = self.Env.reset()
            t_start = self.num_steps
            
            while not done:
                
                action, [val] = self.model(self.sess, np.expand_dims(state, 0))
                new_state, reward, done, info = self.Env.step(action)
                
                totReward += reward
                
                States.append(state)
                Rewards.append(reward)
                Values.append(val)
                Actions.append(action)
                
                if done:
                    new_state = None
                
                self.num_steps += 1
                globalStep += 1
                
                state = np.copy(new_state)
                
                if done or ((self.num_steps - t_start) == self.interval):
                    
                    if done:
                        val = 0.0
                    else:
                        _, [val] = self.model(self.sess, np.expand_dims(state, 0))
                    
                    Returns = self.Discount(Rewards, val)
                    
                    R = np.reshape(Returns, (len(Values),))
                    A = np.reshape(Actions, (len(Returns),))
                    I = np.reshape(States, (len(Actions), 84, 84, 4))
                    self.model.train(self.sess, I, R, A)
                    
                    t_start = self.num_steps
                    
                    Rewards = []
                    States = []
                    Actions = []
                    Values = []
                    
                    self.sess.run(self.model.SyncWithGlobalNetwork)
        
        self.Env.close()
            
    def Discount(self, Rewards, val):
        Returns = []
        R = val
        for i in reversed(range(len(Rewards))):
            R = Rewards[i] + 0.99 * R
            Returns.append(R)
        list.reverse(Returns)
        return Returns