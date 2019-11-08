from tensorforce.environments import Environment
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import time
import random

m = 1.0
g = 9.8
l = 0.2


class SimplePendulumEnv(Environment):
    def __init__(self, visualize=True, n_step_solver=2, dt=1e-2, n_step=1, print_state=False):
        # Here we define the duration of an environment step,
        #                the number of solver steps / environment step
        self.dt = dt
        self.n_step_solver = n_step_solver

        # Number of environemnt steps in an episode
        self.n_step = n_step

        self.print_state = print_state

        # Max action
        self.max_action = 0.7

        # Render of the simulation
        self.visualize = visualize
        if self.visualize:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim([-l, l])
            self.ax.set_ylim([-l, l])

            # We get the cartesian coordinates of the mass
            x = l*np.sin(0)
            y = -l*np.cos(0)

            # We plot the mass
            self.mass, = self.ax.plot(x, y, 'bo', markersize=20)
            # We plot the pole
            self.pole, = self.ax.plot([0, x], [0, y])

            # We show
            plt.show()

        self.y = [0.0, 0.0]

        super(SimplePendulumEnv, self).__init__()

    def actions(self):
        return dict(type='float', shape=(1,), min_value=-self.max_action, max_value=self.max_action)

    def states(self):
        return dict(type='float', shape=(2,))

    def reset(self):
        self.step = 0

        # if random.random() > 0.5:
        #     self.y = [0.0, 0.0]

        self.y = [0.0, 0.0]

        return self.get_state()

    def get_theta(self):
        return (self.y[0] + np.pi) % (2*np.pi) - np.pi

    def get_state(self):
        return (self.get_theta(), self.y[1])

    def execute(self, actions):
        # We process the action
        actions = self.handle_actions(actions)

        # We advance the environment by one step
        self.next_step(actions)

        state = self.get_state()
        theta = self.get_theta()

        # We get the reward
        reward = abs((theta/np.pi))**(2) / self.n_step

        # We visualize
        if self.visualize:
            self.render()

        self.step += 1

        if self.print_state and (self.step % 100 == 0):
            print("step {} theta = {}".format(self.step, theta))
            print("        theta dot: {}".format(self.y[1]))
            print("        moment: {}".format(actions))

        return state, self.step >= self.n_step, reward

    def handle_actions(self, actions):
        return actions[0]
        # return self.max_action

    def next_step(self, actions):
        t = np.linspace(0, self.dt, self.n_step_solver)
        sol = odeint(self.solve_pendulum, self.y, t, args=(actions,))
        self.y = sol[-1]

    @staticmethod
    def solve_pendulum(y, t, C):
        theta, omega = y
        domegadt = -(g/l)*np.sin(theta) + C / (m * l**2)
        dydt = [omega, domegadt]
        return dydt

    def render(self):
        # We get the current value of theta
        theta = self.get_theta()

        # We get the cartesian coordinates of the mass
        x = l*np.sin(theta)
        y = -l*np.cos(theta)

        # We plot the mass
        self.mass.set_data(x, y)
        # We plot the pole
        self.pole.set_data([0, x], [0, y])

        plt.pause(0.0001)

    def close(self):
        if self.visualize:
            plt.close()
