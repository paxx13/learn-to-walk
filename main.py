import os
import argparse
from pathlib import Path
import gym
from general import LinearPolicy, run_episode
from ars import train
import pickle


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='reinforcement learning for the openai gym \'Bipedal Walker\' environment')
    arg_parser.add_argument('-t', '--train', help='set to learn a policy', action="store_true")
    arg_parser.add_argument('--hard', help='set to choose the hard version of the task', action="store_true")
    args = arg_parser.parse_args()

    if args.hard:
        env_name = 'BipedalWalkerHardcore-v2'
    else:
        env_name = 'BipedalWalker-v2'

    dir_name = 'data/'+env_name
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    policy_file = Path(dir_name+'/policy.obj')

    # create environment and set loglevel to 'error')
    env = gym.make(env_name)
    gym.logger.set_level(50)

    if args.train:
        if policy_file.is_file():
            # use pretrained policy to train further
            policy = pickle.load(open(policy_file, "rb"))
        else:
            # create policy from scratch
            policy = None

        p = train(env,policy)
        pickle.dump(p, open(policy_file, "wb" ) )
    else:
        policy = pickle.load(open(policy_file, "rb"))            
        run_episode(env, policy, render=True)