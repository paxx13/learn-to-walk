import numpy as np

class Normalizer():
    # Normalizes the inputs
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
        
    @classmethod 
    def from_param(cls, size, mean, var):
        normalizer = cls(size)
        normalizer.mean = mean
        normalizer.var = var
        return normalizer

    def observe(self, x):
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
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))
        self.normalizer = Normalizer(input_size)
    
    @classmethod 
    def from_param(cls, weights, mean, var):
        policy = cls(weights[0].size, weights[1].size)
        policy.weights = weights
        policy.normalizer = Normalizer.from_param(weights[0].size, mean, var)
        return policy
        
    def get_action(self, state):
        self.normalizer.observe(state)
        self.normalizer.normalize(state)
        return (self.weights).dot(state)

    def update(self, update):
        self.weights += update
        
       
def run_episode(env, policy, render=False):
    done = False
    state = env.reset()
    total_reward = 0
    while not done:
        if render:
            env.render()
        action = policy.get_action(state)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        total_reward += reward
    return total_reward