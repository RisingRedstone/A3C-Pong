#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:25:30 2019

@author: risingredstone
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def get_name(name):
    return re.match('\w*/([^:]*):\w*', name).group(1)

def copy_vars(from_scope, to_scope):
    from_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)

    from_dict = {var.name: var for var in from_tvs}
    to_dict = {var.name: var for var in to_tvs}
    copy_ops = []
    for to_name, to_var in to_dict.items():
        from_name = to_name.replace(to_scope, from_scope)
        from_var = from_dict[from_name]
        op = to_var.assign(from_var.value())
        copy_ops.append(op)

    return copy_ops

def ModelMaker(StateShape, NumAction):
    
    #Here is the Input placeholder which will take the Input values
    Inputs = tf.placeholder(tf.float32, StateShape)
    
    #Here we start defining the model
    conv1 = keras.layers.Conv2D(32, 8,
                                 strides = (4, 4),
                                 padding = 'valid',
                                 activation = 'relu',
                                 name = 'conv1')(Inputs)
    
    conv2 = keras.layers.Conv2D(64, 4,
                                 strides = (2, 2),
                                 padding = 'valid',
                                 activation = 'relu',
                                 name = 'conv2')(conv1)
    
    conv3 = keras.layers.Conv2D(64, 3,
                                 strides = (1, 1),
                                 padding = 'valid',
                                 activation = 'relu',
                                 name = 'conv3')(conv2)
    
    flatten = keras.layers.Flatten()(conv3)
    
    fc1 = keras.layers.Dense(512, activation = 'relu', name = 'Feature1')(flatten)
    
    #Here are the two output layers 
    PolicyLogits = keras.layers.Dense(NumAction, activation = None,
                                     kernel_initializer = normalized_columns_initializer(std = 0.01),
                                     bias_initializer = None,
                                     name = 'policy_logits')(fc1)
    
    #So apparently we need a PolicyLogits and Policy (Probabilities)
    Policy = tf.nn.softmax(PolicyLogits)
    
    #This is the Value output layer
    Value = keras.layers.Dense(1, activation = None,
                                    kernel_initializer = normalized_columns_initializer(std = 1.0),
                                    bias_initializer = None,
                                    name = 'value')(fc1)
    
    Values = Value[:, 0]
    
    layers = [conv1, conv2, conv3, fc1]
    
    return (Inputs, PolicyLogits, Policy, Values, layers)
    
class ActorCritic:
    
    def __init__(self, NumAction, StateShape, ValueCoef, EntropyCoef, scope, device):
        self.scope = scope
        self.NumAction = NumAction
        self.StateShape = StateShape
        self.EntropyCoef = EntropyCoef
        self.ValueCoef = ValueCoef
        
        #Create the input placeholder that will feed into the entire network
        StateShapeList = [None] + list(StateShape)
        self.Actions = tf.placeholder(tf.int32, [None])
        self.Returns = tf.placeholder(tf.float32, [None])
        
        #Put the rest of the variables in a scope and on a device
        with tf.variable_scope(self.scope):
            
            self.Inputs, self.PolicyLogits, self.Policy, self.Values, self.layers = ModelMaker(StateShapeList, self.NumAction)
        
        #Calculate the advantage
        advantage = self.Returns - self.Values
        
        neglogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.PolicyLogits,
                                                                    labels = self.Actions)
        
        '''This is the Entropy method from the code that actually works'''
        logp = self.PolicyLogits - tf.reduce_logsumexp(self.PolicyLogits, axis = -1, keepdims = True)
        nlogp = -logp
        probs = tf.nn.softmax(self.PolicyLogits, axis = -1)
        nplogp = tf.reduce_sum(probs * nlogp, axis = -1, keepdims = True)
        self.Entropy = tf.reduce_mean(nplogp)
        
        #Calculate the Policy Loss and Value Loss and then combine them to create Loss
        self.PolicyLoss = tf.reduce_mean(neglogprob * tf.stop_gradient(advantage))
        self.PolicyLoss = self.PolicyLoss - self.EntropyCoef * self.Entropy
        self.ValueLoss = self.ValueCoef * tf.reduce_mean(0.5 * advantage ** 2)
        
        self.Loss = self.ValueLoss + self.PolicyLoss
        
        #Get the local and global variables
        localvars = tf.trainable_variables(self.scope)
        globalvars = tf.trainable_variables('global')
        
        #Calculate the gradients and then clip them
        grads = tf.gradients(self.Loss, localvars)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        
        #Here is where we define the optimizer...Probably change it later so that the optimizer is defined somewhere else
        self.optimizer = tf.train.RMSPropOptimizer(1e-4, decay = 0.99, epsilon = 1e-5)
        
        '''Here is the gradient update method taken from the same code'''
        computeGradsDict = {}
        for grad, var in zip(grads, localvars):
            if grad is None:
                continue
            var_name = get_name(var.name)
            computeGradsDict[var_name] = grad
            
        globalVarsDict = {}
        for var in globalvars:
            var_name = get_name(var.name)
            globalVarsDict[var_name] = var
        
        gradsAndVars = []
        for var_name, grad in computeGradsDict.items():
            gradsAndVars.append((grad, globalVarsDict[var_name]))
            
        #Here we apply the final gradients to the final trainable variables
        self.ApplyGradients = self.optimizer.apply_gradients(gradsAndVars)
        
        #This code is used to copy global variables to the local variables
        if self.scope != 'global':
            self.SyncWithGlobalNetwork = copy_vars(from_scope='global', to_scope = self.scope)
        
        #This is to save the global variables
        self.Saver = tf.train.Saver(var_list = globalvars)
        
    
    def __call__(self, sess, Inputs):
        '''This function is used to get the value and policy from a state'''
        Pol, Val = sess.run((self.Policy, self.Values), feed_dict = {self.Inputs : Inputs})
        p = np.random.choice(Pol[0], p = Pol[0])
        p = np.argmax(Pol == p)
        return p, Val
    
    def train(self, sess, Inputs, Returns, Actions):
        '''This function is used to train the network'''
        sess.run(self.ApplyGradients, feed_dict = {self.Inputs : Inputs,
                                                   self.Returns : Returns,
                                                   self.Actions : Actions})