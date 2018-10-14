import sys
from sys import argv
import os
from pathlib import Path
import gym
from general import LinearPolicy, run_episode
from ars import train
import numpy as np

    
if __name__ == '__main__':
    if len(argv) < 2:
        print('pass argument \"train\" or \"run\"')
        sys.exit(0)

    env_name = 'BipedalWalker-v2'
    dir_name = 'data/'+env_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    weights_file = Path(dir_name+'/weights.npz')
       
    if os.path.isfile(weights_file):
        data = np.load(weights_file)
    else:
        data = None
        print('!!! no file with pretrained weights found in data folder')
    
    env = gym.make(env_name)
   
    if argv[1] == 'train':
        p = train(env)
        np.savez(weights_file, w=p.weights, mean=p.normalizer.mean, var=p.normalizer.var)
    else:
        policy = LinearPolicy.from_param(data['w'], data['mean'], data['var'])        
        run_episode(env, policy, render=True)