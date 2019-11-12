from tensorforce.agents import Agent
from tensorforce.agents import  DeterministicPolicyGradient
from tensorforce.execution import ParallelRunner
from tensorforce.environments.environment_process_wrapper import ProcessWrapper
# from stable_baselines 
import os
import cProfile

from tensorforce.environments.openai_gym import OpenAIGym
n_env = 4

def build_env(Env, *args, multiprocessing=False, **kwargs):
    # Instantiate the environment
    list_envs = []

    if multiprocessing:
        for _ in range(n_env):
            list_envs.append(ProcessWrapper(Env(*args, **kwargs)))
    else:
        for _ in range(n_env):
            list_envs.append(Env(*args, **kwargs))

    return list_envs

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


def base_test(env):
    batch_size = 20
    n_step = 2000

    agent = Agent.create(
        agent='ppo',
        batch_size=batch_size,
        learning_rate=1e-3,
        states=env[0].states(),
        actions=env[0].actions(),
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
    runner = ParallelRunner(agent=agent, environments=env)

    # Start the runner
    runner.run(num_episodes=100, sync_episodes=False)
    runner.close()

def test_cartpole(multiprocessing=False):
    env = build_env(OpenAIGym, 'CartPole-v0', multiprocessing=multiprocessing, visualize=False)
    base_test(env)

pr = cProfile.Profile()
pr.enable()
test_cartpole()
pr.disable()
pr.print_stats(sort='time')

pr2 = cProfile.Profile()
pr2.enable()
test_cartpole(multiprocessing=True)
pr2.disable()
pr2.print_stats(sort='time')