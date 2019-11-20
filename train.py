#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:44:14 2019

@author: risingredstone
"""

from environment import EnvironmentProcess
from model import ModelMaker, ActorCritic
from worker import Worker
import tensorflow as tf
import threading
import numpy as np
from pynput import keyboard
import time
from arguments import Arguments

args = Arguments()
UpdatePerSteps =  args.stepsperupdate
TotalSteps = args.n_steps
EnvNums = args.n_workers
EnvName = args.env_id
device = '/device:GPU:0'
filePath = args.savedir

SaveModelFlag = False

with tf.variable_scope('global'):
    ModelMaker([None] + list((84, 84, 4)), 6)

def on_release(key):
    try:
        if key.char == 's':
            global SaveModelFlag
            print("Saving")
            SaveModelFlag = True
    except:
        pass
    
def train(sess):
    
    threads = []
    for i in range(EnvNums):
        Name = 'Env' + str(i)
        env = EnvironmentProcess(EnvName, False, (84, 84), 4)
        model = ActorCritic(6, (84, 84, 4), args.valuecoef, args.entropycoef, Name, device)
        work = Worker(sess, env, model, TotalSteps, UpdatePerSteps, Name)
        t = threading.Thread(target = work.run)
        threads.append(t)
    
    sess.run(tf.global_variables_initializer())
    
    Saver = tf.train.Saver(var_list = tf.trainable_variables('global'))
    global filePath
    
#    try:
#        Saver.restore(sess, filePath + 'final.ckpt')
#    except:
#        print("File Not Loaded")
        
    listener = keyboard.Listener(
                on_release=on_release)
    listener.start()
    
    for t in threads:
        t.start()
    
    global SaveModelFlag
    while True:
        if SaveModelFlag:
            Saver.save(sess, filePath + 'final.ckpt')
            SaveModelFlag = False
            print("Saved")
        time.sleep(10)
    
    for t in threads:
        t.join()

def test(sess):
    GlobalModel = ActorCritic(6, (84, 84, 4), args.valuecoef, args.entropycoef, 'test', device)
    
    #sess.run(tf.global_variables_initializer())
    
    Saver = tf.train.Saver(var_list = tf.trainable_variables('global'))
    try:
        Saver.restore(sess, filePath + 'final.ckpt')
    except:
        print("File Not Loaded")
    sess.run(GlobalModel.SyncWithGlobalNetwork)
    env = EnvironmentProcess(EnvName, True, (84, 84), 4)
    
    while True:
        totReward = 0
        obs = env.reset()
        done = False
        while not done:
            action, _ = GlobalModel(sess, np.expand_dims(obs, 0))
            obs, reward, done, _ = env.step(action)
            totReward += reward
            time.sleep(1/120)
        print("This is the total reward : {}".format(totReward))
    env.close()

testing = args.testing
if __name__ == '__main__':
    with tf.Session() as sess:
        if not testing:
            train(sess)
        else:
            test(sess)
