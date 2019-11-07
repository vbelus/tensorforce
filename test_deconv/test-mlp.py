from tensorforce.agents import Agent
from tensorforce.agents import  DeterministicPolicyGradient
from env_mlp import SingleValueEnv
from tensorforce.execution import Runner

batch_size = 20

# Instantiate the environment
env = SingleValueEnv()

actor_network = [
    dict(type='dense', size=64, activation='relu'),
    dict(type='dense', size=64, activation='relu')
]

critic_network = [
    dict(type='dense', size=64, activation='relu'),
    dict(type='dense', size=64, activation='relu')
]

agent = Agent.create(
    agent='ppo',
    states=env.states(),
    actions=env.actions(),
    network=actor_network,
    discount=0.99,
    entropy_regularization=None,
    critic_network=critic_network,
    critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
    max_episode_timesteps=10
)

# Initialize the runner
runner = Runner(agent=agent, environment=env)

# Start the runner
runner.run(num_episodes=30000)
runner.close()