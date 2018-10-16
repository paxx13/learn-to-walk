import numpy as np
from tqdm import trange
from general import LinearPolicy, run_episode
from multiprocessing import Pool
import multiprocessing
from functools import partial
import gym


N_DIR = 20
N_BEST = 10
assert(N_BEST<=N_DIR)
SIGMA_EXPLORE_NOISE = 0.05
ALPHA = 0.02
TRAIN_STEPS = 1000
NUM_WORKERS = multiprocessing.cpu_count()*2


def rollout(delta, env, policy):
    policy_p = LinearPolicy.from_param(policy.weights + SIGMA_EXPLORE_NOISE*delta, policy.env_stats)
    policy_n = LinearPolicy.from_param(policy.weights - SIGMA_EXPLORE_NOISE*delta, policy.env_stats)

    # run episodes and collect rewards
    r_pos, s_pos = run_episode(env, policy_p)
    r_neg, s_neg = run_episode(env, policy_n)

    return r_pos, r_neg, np.append(s_pos, s_neg, axis=0)


# Collect 2N rollouts and their corresponding rewards using the 2N policies
def collect_rollouts(p, env, policy, deltas):
    rollout_deltas=partial(rollout, env=env, policy=policy)
    rollouts = p.map(rollout_deltas, deltas)

    rpos = []
    rneg = []
    for r in rollouts:
        rpos.append(r[0])
        rneg.append(r[1])
        # update environment statistics
        for step in r[2]:
            policy.env_stats.update(step)

    return [rpos, rneg]


# augmented random search
# https://arxiv.org/pdf/1803.07055.pdf
def train(env, policy=None):     
    if policy is None:
        policy = LinearPolicy(env.observation_space.shape[0],env.action_space.shape[0])
        
    p = Pool(NUM_WORKERS) 

    train_steps = trange(TRAIN_STEPS)

    for s in train_steps:
        gym.logger.set_level(40)
        deltas = np.array([np.random.randn(*policy.weights.shape) for _ in range(N_DIR)])

        # collect rewards using the random directions
        rollouts = collect_rollouts(p, env, policy, deltas)

        # sort the directions by experienced rewards during rollout and choose the best
        order = np.argsort(-np.amax(rollouts, axis=0))
        order = order[0:N_BEST]

        deltas = deltas[order]
        r = np.array(rollouts)
        r = r[:,order]

        # calculate the update step
        sigma_rewards = np.sum(r, axis=0).std()

        sum = np.sum([ (r[0][i] - r[1][i])*deltas[i] for i in range(len(r)) ], axis=0)

        update = ALPHA / (N_DIR * sigma_rewards) * sum

        # update policy 
        policy.update(update)

        # evaluate score
        score, _ = run_episode(env, policy)  
        train_steps.set_description('score: %.2f' % score)
    return policy
