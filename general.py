import numpy as np
import gym

MAX_STEPS=5000

class EnvironemntStats():
    # collects environment statistics
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.full(nb_inputs,1e-2)

    def update(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class LinearPolicy():
    # linear policy normalizing environment observations
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))
        self.env_stats = EnvironemntStats(input_size)

    @classmethod 
    def from_param(cls, weights, env_stats):
        policy = cls(weights[0].size, weights[1].size)
        policy.weights = weights
        policy.env_stats = env_stats
        return policy

    def get_action(self, state):
        state = self.env_stats.normalize(state)
        return (self.weights).dot(state)

    def update(self, update):
        self.weights += update


def run_episode(env, policy, render=False):
    # set log level to 'error'
    gym.logger.set_level(40)
    state = np.array([env.reset()])
    total_reward = 0
    for s in range(MAX_STEPS):
        if render:
            env.render()

        action = policy.get_action(state[-1])
        s, reward, done, _ = env.step(action)
        state = np.append(state, [s], axis=0)
        reward = np.clip(reward, -10, 10)
        total_reward += reward
        if done:
            break;

    return total_reward, state