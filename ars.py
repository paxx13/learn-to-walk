# https://arxiv.org/pdf/1803.07055.pdf
import numpy as np
from tqdm import trange
from general import LinearPolicy,run_episode

N_DIR = 20
N_BEST = 10
assert(N_BEST<=N_DIR)
SIGMA_EXPLORE_NOISE = 0.03
ALPHA = 0.02
TRAIN_STEPS = 5

        
# Collect 2N rollouts of horizon H and their corresponding rewards using the 2N policies
def collect_rollouts(env, policy, deltas):
    r_pos = []
    r_neg = []
    for n in range(N_DIR):
        # create policies with method of finite differences
        policy_p = LinearPolicy.from_param(policy.weights + SIGMA_EXPLORE_NOISE*deltas[n], policy.normalizer.mean, policy.normalizer.var)
        policy_n = LinearPolicy.from_param(policy.weights - SIGMA_EXPLORE_NOISE*deltas[n], policy.normalizer.mean, policy.normalizer.var)
         
        # run episodes and collect rewards
        r_pos.append(run_episode(env, policy_p))
        r_neg.append(run_episode(env, policy_n))
    return (r_pos, r_neg)
    

# augmented random search
def train(env, weights=None, mean=None, var=None):     
    if weights is not None:
        policy = LinearPolicy.from_param(weights, mean, var)
    else:
        policy = LinearPolicy(env.observation_space.shape[0],env.action_space.shape[0])
        
    train_steps = trange(TRAIN_STEPS)
    for s in train_steps:
        deltas = np.array([np.random.randn(*policy.weights.shape) for _ in range(N_DIR)])
        
        # collect rewards using the random directions
        rollouts = collect_rollouts(env, policy, deltas)

        # sort the directions by their experienced rewards and choose the best
        order = np.array(np.argsort(-np.amax(rollouts, axis=0)))
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
        score = run_episode(env, policy)  
        train_steps.set_description('score: %.2f' % score)
    return policy
