from tensorforce.environments import Environment
import numpy as np
import time


class SingleValueEnv(Environment):
    def __init__(self):
        self.n_step = 1

        self.value = 0.0

        super(SingleValueEnv, self).__init__()

    def actions(self):
        return dict(type='float', shape=(1,), min_value=0.0, max_value=1.0)

    def states(self):
        return dict(type='float', shape=(1,))

    def reset(self):
        self.step = 0
        
        self.value = np.random.random_sample((1,)) * 2 - 1

        return self.value

    def get_state(self):
        return (self.value / 2 + 0.5)

    def execute(self, actions=None):
        actions = self.handle_actions(actions)

        self.value = np.clip((self.value + actions), -1, 1)

        reward = (1 - abs(self.value)) / self.n_step

        self.step += 1

        return self.get_state(), self.step>=self.n_step, reward

    def handle_actions(self, actions):
        actions = np.array(actions)
        actions = actions * 2 - 1
        return actions
