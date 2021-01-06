from typing import Type, Callable, List, Dict, Tuple, Optional, Any
from lucy.utils.helper import gumbel_softmax

import gym
import numpy as np

import torch
from torch import nn

from stable_baselines3.dqn.policies import BasePolicy, FlattenExtractor, BaseFeaturesExtractor

AutoEncoder = Any




class CustomModelQNetwork(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, observation: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # return self.forward(observation)  # [B, V]
        logits = self.forward(observation)  # [B, V]
        action = self.sample(logits)
        return action


class CNNBasedQNetwork(CustomModelQNetwork):
    """
    Here we implement neural network to do all computing from observation space to action space.
    Must implement a forward function, everything else is done in the other classes.
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 pretrained_model: AutoEncoder,
                 pad_index: int):
        super().__init__(observation_space=observation_space, action_space=action_space)
        self.pretrained_model = pretrained_model
        self.pad_index = pad_index

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B, _, _ = obs.shape
        try:
            t = int(torch.where(obs[0, :, self.pad_index] == 1.)[0][0])
        except IndexError:
            t = obs.shape[1]

        x = obs.unsqueeze(1)
        h = self.pretrained_model.decoder.layers(x)  # [B, D', T, 1]
        h = torch.squeeze(h, 3).permute(0, 2, 1)  # [B, T, D']
        logits = self.pretrained_model.decoder.hidden_2_logits(h)  # [B, T, V]
        logits = logits[:, t-1, :]
        return logits

        # sample_prob = torch.nn.functional.softmax(logits, dim=-1).squeeze(1)  # [B, V]
        # action = torch.multinomial(sample_prob, num_samples=1)  # [B, 1]
        #
        # action = action.view(-1)
        # return action
        # # return logits

    def sample(self, logits: torch.Tensor, deterministic: bool = False) -> int:
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            sample_prob = torch.nn.functional.softmax(logits, dim=-1).squeeze(1)  # [B, V]
            action = torch.multinomial(sample_prob, num_samples=1)  # [B, 1]
        action = action.view(-1)
        return action


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
            pretrained_model: AutoEncoder = None,
            pad_index: int = 0,
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

        self.pretrained_model = pretrained_model
        self.pad_index = pad_index
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

    def make_q_net(self) -> CustomModelQNetwork:
        return CNNBasedQNetwork(self.observation_space, self.action_space, self.pretrained_model, pad_index=self.pad_index)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.q_net.predict(obs, deterministic=deterministic)

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

