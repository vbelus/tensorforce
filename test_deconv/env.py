from tensorforce.environments import Environment
import numpy as np
import time


class MatrixEnv(Environment):
    def __init__(self, length_matrix):
        self.shape = (length_matrix, length_matrix, 1)
        self.n_step = 1
        self.length_matrix = length_matrix

        self.matrix = np.zeros((self.shape))

        super(MatrixEnv, self).__init__()

    def actions(self):
        return dict(type='float', shape=self.shape, min_value=0.0, max_value=1.0)

    def states(self):
        return dict(type='float', shape=self.shape)

    def reset(self):
        self.step = 0

        self.matrix = np.random.random_sample(self.shape) * 2 - 1

        return self.matrix

    def get_state(self):
        return (self.matrix / 2 + 0.5)

    def execute(self, actions):
        actions = self.handle_actions(actions)

        self.matrix = np.clip((self.matrix + actions), -1, 1)

        reward = np.sum((1 - abs(self.matrix) / self.n_step)) / self.length_matrix**2

        self.step += 1

        return self.get_state(), self.step>=self.n_step, reward

    def handle_actions(self, actions):
        actions = np.array(actions)
        actions = actions * 2 - 1
        return actions
