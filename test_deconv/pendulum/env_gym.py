import gym
from gym import spaces
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import time
import random

m = 1.0
g = 9.8
l = 0.2


class SimplePendulumEnv(gym.Env):
    def __init__(self, visualize=False, n_step_solver=2, dt=1e-2, n_step=1, print_state=False):
        # Here we define the duration of an environment step,
        #                the number of solver steps / environment step
        self.dt = dt
        self.n_step_solver = n_step_solver

        # Number of environemnt steps in an episode
        self.n_step = n_step

        self.print_state = print_state

        # Max action
        self.max_action = 2.0

        # Render of the simulation
        self.visualize = visualize
        if self.visualize:
            plt.ion()
            self.fig = plt.figure()

            self.ax = self.fig.add_subplot(211)
            self.ax.set_xlim([-l, l])
            self.ax.set_ylim([-l, l])

            self.ax2 = self.fig.add_subplot(212)
            self.ax2.set_xlim([0, n_step])
            self.ax2.set_ylim([-1, 1])

            # We get the cartesian coordinates of the mass
            x = l*np.sin(0)
            y = -l*np.cos(0)

            # We plot the mass
            self.mass, = self.ax.plot(x, y, 'bo', markersize=20)
            # We plot the pole
            self.pole, = self.ax.plot([0, x], [0, y])

            self.data = np.zeros((3, self.n_step))
            self.theta_plot, = self.ax2.plot()

            # We show
            plt.show()

        self.y = [0.0, 0.0]

        # Define action space and state space
        self.action_space = self.actions()
        self.observation_space = self.states()

        super(SimplePendulumEnv, self).__init__()

    def actions(self):
        high = np.array([
            self.max_action
        ])
        return spaces.Box(-high, high, dtype=np.float32)

    def states(self):
        # return spaces.Dict({
        #     'x_position': spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        #     'y_position': spaces.Box(low=-1.0, high=1.0, shape=(1,)),
        #     'omega': spaces.Box(low=np.inf, high=-np.inf, shape=(1,))
        # })

        high = np.array([
            1,
            1,
            np.finfo(np.float32).max,
        ])
        return spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        self._step = 0

        # if random.random() > 0.5:
        #     self.y = [0.0, 0.0]

        self.y = [0.0, 0.0]

        return self.get_state()

    def get_theta(self):
        return (self.y[0] + np.pi) % (2*np.pi) - np.pi

    def get_state(self):
        theta = self.get_theta()

        # We get the cartesian coordinates of the pole
        x = np.sin(theta)
        y = np.cos(theta)

        # state = {
        #     'x_position': np.array(x),
        #     'y_position': np.array(y),
        #     'omega': np.array(self.y[1])
        # }

        state = np.array([
            x,
            y,
            self.y[1]
        ])

        return state

    def step(self, actions):
        # We process the action
        actions = self.handle_actions(actions)

        # We advance the environment by one step
        self.next_step(actions)

        state = self.get_state()
        theta = self.get_theta()

        # We get the reward
        reward = (10 * abs((theta/np.pi))**(2) - 0.002 * self.y[1]**2 - 0.001*actions) / self.n_step

        # We visualize
        if self.visualize:
            self.render()

        self._step += 1

        if self.print_state and (self._step % 100 == 0):
            print("step {} theta = {}".format(self._step, theta))
            print("        theta dot: {}".format(self.y[1]))
            print("        moment: {}".format(actions))

        return state, self._step >= self.n_step, reward

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
