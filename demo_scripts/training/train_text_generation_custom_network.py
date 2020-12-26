from typing import Type, Callable, List, Dict, Tuple, Optional, Any

from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.reward import CounterScore

import numpy as np
from rich import print

import gym


import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork, BasePolicy, FlattenExtractor, BaseFeaturesExtractor
from stable_baselines3 import DQN


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
    print(f"Generated {''.join(actions)}")
    print(f"Total Reward: {total_reward}")
    print("---------------------------------------------")


class CustomQNetwork(BasePolicy):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space):
        super().__init__(observation_space,
                         action_space)
        self.q_net = QNET(self.observation_space, self.action_space)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs.shape: (batch_size,) + self.observation_space.shape
        return self.q_net(obs)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class QNET(nn.Module):
    """
    Here we implement neural network to do all computing from observation space to action space.
    Must implement a forward function, everything else is done in the other classes.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.flat_dim = int(np.prod(observation_space.shape))
        self.out_dim = action_space.n
        self.linear = nn.Linear(self.flat_dim, self.out_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.view(-1, self.flat_dim)
        return self.linear(obs)

"""
Policy initialized in DQN with

self.policy = self.policy_class(
    self.observation_space,
    self.action_space,
    self.lr_schedule,
    **self.policy_kwargs  # pytype:disable=not-instantiable
)

where self.policy_class is DQNPolicy
"""


class CustomDQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        """
        Create the network and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> CustomQNetwork:
        return CustomQNetwork(self.observation_space, self.action_space)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data


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

model = DQN(env=env, policy=CustomDQNPolicy, gamma=0.99, batch_size=32, learning_rate=5e-4,
            exploration_fraction=0.1, verbose=1)

for i in range(int(1000)):
    if i % 10 == 0:
        model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False, log_interval=None)
        eval_model(model, env, epoch=i)
    else:
        model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False, log_interval=None)

