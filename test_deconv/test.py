from tensorforce.agents import Agent
from tensorforce.agents import  DeterministicPolicyGradient
from env import MatrixEnv
from tensorforce.execution import Runner

length_matrix = 9
batch_size = 20

# Instantiate the environment
env = MatrixEnv(length_matrix)

actor_network = [
    dict(type='conv2d', size=64, window=1, activation='relu', is_trainable=True),
    dict(type='conv2d', size=64, window=1, activation='relu', is_trainable=True)
]

critic_network = [
    dict(type='conv2d', size=64, window=1, activation='relu', is_trainable=True),
    dict(type='conv2d', size=64, window=1, activation='relu', is_trainable=True)
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


# agent = Agent.create(
#         agent='ppo', environment=env,
#         max_episode_timesteps=40,
 
#         network=network,
#         # Optimization
#         batch_size=10, update_frequency=10, learning_rate=1e-3, subsampling_fraction=0.2,
#         optimization_steps=5,
#         # Reward estimation
#         likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
#         # Critic
#         critic_network=critic_network,
#         critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
#         # Preprocessing
#         preprocessing=None,
#         # TensorFlow etc
#         name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
#         summarizer=None, recorder=None
#     )


# # Instantiate a Tensorforce agent
# agent = Agent.create(
#     agent='tensorforce',
#     states=env.states(),
#     actions=env.actions(),
#     max_episode_timesteps=20,
#     memory=10000,
#     update=dict(unit='timesteps', batch_size=64),
#     optimizer=dict(type='adam', learning_rate=3e-4),
#     policy=dict(network=network),
#     objective='policy_gradient',
#     reward_estimation=dict(horizon=20)
# )