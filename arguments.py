#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 19:05:42 2019

@author: risingredstone
"""

import argparse

def Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("--testing", type=bool, default = False)
    parser.add_argument("--n_steps", type=float, default = 80e6)
    parser.add_argument("--n_workers", type=int, default = 1)
    parser.add_argument("--savedir", default = 'Saves/network')
    
    #Training HyperParameters
    parser.add_argument("--stepsperupdate", type=int, default = 5)
    parser.add_argument("--valuecoef", type=float, default = 0.5)
    parser.add_argument("--entropycoef", type=float, default = 0.01)
    
    args = parser.parse_args()
    
    return args