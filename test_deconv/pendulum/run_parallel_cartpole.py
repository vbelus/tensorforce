from tensorforce.agents import Agent
from tensorforce.agents import  DeterministicPolicyGradient
from tensorforce.environments.openai_gym import OpenAIGym
from tensorforce.environments.environment import Environment

from tensorforce.execution import ParallelRunner
from tensorforce.execution import Runner

import os

batch_size = 20
n_step = 2000

# Instantiate the environment
n_env = 15

list_envs = []

env = Environment.create(environment='gym', level='CartPole-v0')
# list_envs.append(env)

# for index in range(n_env-1):
#     # list_envs.append(ProcessWrapper(OpenAIGym('CartPole-v0', visualize=False)))
#     list_envs.append(OpenAIGym('CartPole-v0', visualize=False))


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
    agent='ppo',
    batch_size=batch_size,
    learning_rate=1e-3,
    states=env.states(),
    actions=env.actions(),
    network=actor_network,
    discount=1.0,
    entropy_regularization=None,
    critic_network=critic_network,
    critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
    max_episode_timesteps=n_step,
    parallel_interactions=n_env,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=30)
)

agent.initialize()

# Initialize the runner
# runner = ParallelRunner(agent=agent, environments=list_envs)
runner = Runner(agent=agent, environment=env)

# Start the runner
runner.run(num_episodes=30000)
runner.close()