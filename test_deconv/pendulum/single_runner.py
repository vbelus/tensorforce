from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner

from env import SimplePendulumEnv
import os

n_step = 2000

env = SimplePendulumEnv(visualize=True, n_step=n_step, print_state=True)


actor_network = [
    dict(type='dense', size=128, activation='relu'),
    dict(type='dense', size=64, activation='relu'),
    dict(type='dense', size=64, activation='relu')
]

critic_network = [
    dict(type='dense', size=128, activation='relu'),
    dict(type='dense', size=64, activation='relu'),
    dict(type='dense', size=64, activation='relu')
]


agent = Agent.create(
    agent='dueling_dqn',
    batch_size=10,
    learning_rate=1e-3,
    states=env.states(),
    actions=env.actions(),
    network=actor_network,
    discount=0.99,
    entropy_regularization=None,
    critic_network=critic_network,
    critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
    max_episode_timesteps=n_step,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'))
)

agent.initialize()

state = env.reset()

for k in range(1 * n_step):
    #environment.print_state()
    action = agent.act(state, deterministic=True, independent=True)
    print(action)
    state, terminal, reward = env.execute(action)
