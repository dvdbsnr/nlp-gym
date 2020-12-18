from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.reward import CounterScore

import numpy as np
from rich import print

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3 import DQN


def eval_model(model, env):
    done = False
    obs = env.reset()
    if obs.shape == env.observation_space.shape:
        obs = np.expand_dims(obs, 0)
    total_reward = 0.0
    actions = []
    while not done:
        action, _states = model.predict(obs)
        if isinstance(action, np.ndarray):
            try:
                action = action.item()
            except ValueError:
                print(action)
                raise
        obs, rewards, done, info = env.step(action)
        if obs.shape == env.observation_space.shape:
            obs = np.expand_dims(obs, 0)
        actions.append(env.action_space.ix_to_action(action))
        total_reward += rewards
    print("---------------------------------------------")
    print(f"Generated {''.join(actions)}")
    print(f"Total Reward: {total_reward}")
    print("---------------------------------------------")


# reward function
reward_fn = CounterScore(string_to_count='ab')

vocab = ['SOS', 'EOS'] + list('abcdefg')

env = TextGenEnv(vocabulary=vocab,
                 max_steps=20,
                 reward_function=reward_fn,
                 latent_dim=2,
                 observation_featurizer=None,
                 SOS=vocab.index('SOS'),
                 EOS=vocab.index('EOS'),
                 return_obs_as_vector=True)

# check the environment
check_env(env, warn=True)

model = DQN(env=env, policy=DQNPolicy, gamma=0.99, batch_size=32, learning_rate=5e-4,
            exploration_fraction=0.1, verbose=1)

for i in range(int(1000)):
    if i % 10 == 0:
        print(f'Epoch: {i}')
        model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False, log_interval=2000)
        eval_model(model, env)
    else:
        model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False, log_interval=2000)

