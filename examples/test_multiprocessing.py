from tensorforce.agents import Agent
from tensorforce.agents import  DeterministicPolicyGradient
from tensorforce.execution import ParallelRunner
from tensorforce.environments.environment_process_wrapper import ProcessWrapper
# from stable_baselines 
import os
import cProfile, pstats

from tensorforce.environments.openai_gym import OpenAIGym
from tensorforce.environments import Environment
from gym_film.envs.film_env import FilmEnv
n_env = 16
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
    dict(type='conv2d', size=0, window=1, activation='relu', is_trainable=True)
]

# actor_network = [
#     dict(type='dense', size=128, activation='relu'),
#     dict(type='dense', size=64, activation='relu'),
#     dict(type='dense', size=64, activation='relu')
# ]

critic_network = [
    dict(type='conv2d', size=0, window=1, activation='relu', is_trainable=True)
]

n_step = 400
def base_test(env):
    batch_size = 24

    agent = Agent.create(
        agent='ppo',
        environment=env[0],
        batch_size=batch_size,
        learning_rate=1e-3,
        network=actor_network,
        discount=1.0,
        entropy_regularization=None,
        critic_network=critic_network,
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        max_episode_timesteps=n_step,
        parallel_interactions=n_env
        # saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=30)
    )

    agent.initialize()

    # Initialize the runner
    runner = ParallelRunner(agent=agent, environments=env)

    # Start the runner
    runner.run(num_episodes=48)
    runner.close()

def test_cartpole(multiprocessing=False):
    env = build_env(OpenAIGym, 'CartPole-v0', multiprocessing=multiprocessing, visualize=False)
    base_test(env)

# Not working (Python version ?)
def test_maze(multiprocessing=False):
    env = build_env(Environment.create, 'mazeexp', level=0, multiprocessing=multiprocessing)
    base_test(env)

def test_fluid_film(multiprocessing=False):
    # Instantiate the environment
    list_envs = []

    if multiprocessing:
        for _ in range(n_env):
            list_envs.append(ProcessWrapper(OpenAIGym(FilmEnv(render=False), max_episode_timesteps=n_step)))
    else:
        for _ in range(n_env):
            list_envs.append(OpenAIGym(FilmEnv(render=False), max_episode_timesteps=n_step))

    base_test(list_envs)

###############################################
pr = cProfile.Profile()
pr.enable()

# test_cartpole()
# test_maze()
test_fluid_film()

pr.disable()
ps = pstats.Stats(pr).sort_stats('time')
ps.print_stats(10)

###############################################
pr2 = cProfile.Profile()
pr2.enable()

# test_cartpole(multiprocessing=True)
# test_maze(multiprocessing=True)
test_fluid_film(multiprocessing=True)

pr2.disable()
ps = pstats.Stats(pr2).sort_stats('time')
ps.print_stats(10)