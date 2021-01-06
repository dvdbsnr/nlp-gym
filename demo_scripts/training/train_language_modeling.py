from typing import List, Tuple, Dict

from nlp_gym.data_pools.custom_language_modeling_pool import AAPDDataPool
from nlp_gym.envs.language_modeling.env import LanguageModelingEnv
from nlp_gym.metrics.multi_label import F1Score
from stable_baselines3.dqn.policies import MlpPolicy as DQNPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from rich import print

import numpy as np


def eval_model(model, env, epoch: int):
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
    print(f"Epoch {epoch}")
    env.render()
    print("---------------------------------------------")


# data pool
pool = AAPDDataPool.prepare(split="train")
vocab: List[str] = pool.get_vocabulary()
vocab += ['SOS', 'EOS', 'PAD', 'UNK']

# reward function
reward_fn = None #F1Score()

# multi label env
env = LanguageModelingEnv(vocabulary=vocab,
                          add_sos_eos=True,
                          reward_function=reward_fn,
                          return_obs_as_vector=True)
for sample, weight in pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)

# train a MLP Policy
model = DQN(env=env,
            policy=DQNPolicy,
            gamma=0.99,
            batch_size=4,
            learning_rate=1e-3,
            exploration_fraction=0.1,
            # policy_kwargs={"layers": [16]},
            verbose=1)

for i in range(int(1e+2)):
    model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False, log_interval=1000)
    eval_model(model, env, i)
