from tensorforce.agents import Agent
from tensorforce.agents import  DeterministicPolicyGradient
from env import SimplePendulumEnv
from tensorforce.execution import ParallelRunner
from tensorforce.environments.environment_process_wrapper import ProcessWrapper
# from stable_baselines 
import os

env = ProcessWrapper(SimplePendulumEnv(n_step=200, visualize=False))
print(env.states())
print("a is: ", env.test(a=2))
print("self.thread is: ", env.thread)
print("self.y is: ", env.y())
print("exiting")
import sys
sys.exit()

batch_size = 20
n_step = 2000

# Instantiate the environment
n_env = 15

list_envs = []

env = ProcessWrapper(SimplePendulumEnv(visualize=False, n_step=n_step, print_state=True))
list_envs.append(env)

for index in range(n_env-1):
    list_envs.append(ProcessWrapper(SimplePendulumEnv(n_step=n_step, visualize=False)))


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
runner = ParallelRunner(agent=agent, environments=list_envs)

# Start the runner
runner.run(num_episodes=30000, sync_episodes=False)
runner.close()