from tensorforce.environments import Environment
import numpy as np


class MatrixEnv(Environment):
    def __init__(self, length_matrix):
        self.shape = (length_matrix, length_matrix, 1)
        self.n_step = 20
        self.length_matrix = length_matrix

        self.matrix = np.zeros((self.shape))

        super(MatrixEnv, self).__init__()

    def actions(self):
        return dict(type='float', shape=self.shape)

    def states(self):
        return dict(type='float', shape=self.shape)

    def reset(self):
        self.step = 0
        self.matrix = np.random.random_sample(self.shape) * 2 - 1

        return self.matrix

    def get_state(self):
        return self.matrix

    def execute(self, actions=None):
        actions = self.handle_actions(actions)
        # print(actions[0,0])

        self.matrix = np.tanh(self.matrix + actions)

        reward = 1 - np.sum(abs(self.matrix)) / (self.length_matrix**2)

        self.step += 1

        return self.matrix, self.step>self.n_step, reward

    def handle_actions(self, actions):
        # assert len(actions.shape) == 2
        # assert actions.shape == self.matrix.shape
        actions = np.array(actions)
        return actions
