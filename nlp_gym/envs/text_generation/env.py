import copy
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.base_env import BaseEnv
from nlp_gym.envs.text_generation.observation import Observation
from nlp_gym.envs.text_generation.featurizer import OneHotFeaturizerForTextGeneration
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer
from rich import print
from gym import spaces


@dataclass(init=True)
class DataPoint:
    observation: Observation


class TextGenEnv(BaseEnv):

    def __init__(self,
                 vocabulary: List[str],
                 max_steps: int,
                 reward_function: RewardFunction,
                 observation_featurizer: BaseObservationFeaturizer = None,
                 latent_dim: int = None,
                 SOS: int = 0,
                 EOS: int = 1,
                 PAD: int = 2,
                 return_obs_as_vector: bool = True):
        self.time_step: Optional[int] = None
        self.current_sample: Optional[DataPoint] = None

        self.SOS: int = SOS
        self.EOS: int = EOS
        self.PAD: int = PAD

        # set action and observation spaces
        self.action_space = self._get_action_space(vocabulary)

        if observation_featurizer is None and return_obs_as_vector:
            observation_featurizer = OneHotFeaturizerForTextGeneration(self.action_space, latent_dim=latent_dim)

        # max steps is max sequence length
        super().__init__(max_steps=max_steps,
                         reward_function=reward_function,
                         observation_featurizer=observation_featurizer,
                         return_obs_as_vector=return_obs_as_vector)

    @staticmethod
    def _get_action_space(vocabulary: List[str]) -> ActionSpace:
        actions = copy.deepcopy(vocabulary)
        action_space = ActionSpace(actions)
        return action_space

    def step(self, action: int) -> Tuple[Union[Observation, np.array], float, bool, dict]:
        action_str = self.action_space.ix_to_action(action)

        self.time_step += 1

        # later:
        # if done:
        #   step_reward = self.reward_function(XXX)
        # else:
        #   step_reward = 0.
        step_reward = self.reward_function(self.current_sample.observation, action_str, None)

        done = self.time_step >= self.max_steps or action == self.EOS
        if not done:
            updated_observation = self.current_sample.observation.get_updated_observation(action_str,
                                                                                          self.time_step,
                                                                                          self.observation_featurizer,
                                                                                          self.return_obs_as_vector)

            # update the current sample (just the observation)
            self.current_sample.observation = updated_observation
        else:
            self.current_sample.observation.current_action_history.append(action_str)

        # return observation, reward, done, info
        observation_to_return = self.current_sample.observation.get_vector().numpy() if self.return_obs_as_vector \
            else self.current_sample.observation
        return observation_to_return, step_reward, done, {}

    def reset(self, sample: Sample = None) -> Union[BaseObservation, np.array]:
        if sample is not None:
            raise NotImplementedError('We dont need datapoints here.')

        prior_mean = torch.nn.Parameter(torch.zeros(self.observation_featurizer.latent_dim), requires_grad=False)
        prior_var = torch.nn.Parameter(torch.ones(self.observation_featurizer.latent_dim), requires_grad=False)
        prior = torch.distributions.Normal(prior_mean, prior_var)

        latent_vector = prior.sample()
        # latent_vector = latent_vector.fill_(0.)

        self.time_step = 0

        # get initial observation
        observation = Observation.build(self.time_step,
                                        latent_vector=latent_vector,
                                        max_length=self.max_steps,
                                        action_history=[self.action_space.ix_to_action(self.SOS)],
                                        observation_featurizer=self.observation_featurizer,
                                        featurize=self.return_obs_as_vector)

        # construct current data point
        self.current_sample = DataPoint(observation=observation)

        # return observation
        if self.return_obs_as_vector:
            observation_to_return = self.current_sample.observation.get_vector().numpy()
            # if observation_to_return.shape == self.observation_space.shape:
            #     # observation_to_return = np.expand_dims(observation_to_return, axis=0)
        else:
            observation_to_return = self.current_sample.observation
        return observation_to_return

    def close(self):
        pass

    def render(self):
        """
        Renders the current state of the environment
        """
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        history = [""] if self.time_step == 0 else self.current_sample.observation.get_current_action_history()
        for action in history:
            print(f"[italic red]Generated[/italic red]:{action}")

    def add_sample(self, sample: Sample, weight: int = 1.0):
        raise NotImplementedError

    def get_samples(self) -> List[Sample]:
        raise NotImplementedError

    def _set_spaces(self, observation_featurizer: BaseObservationFeaturizer):
        low = np.full(shape=(self.max_steps, observation_featurizer.get_observation_dim(),),
                      fill_value=-float('inf'),
                      dtype=np.float32)
        high = np.full(shape=(self.max_steps, observation_featurizer.get_observation_dim(),),
                       fill_value=float('inf'),
                       dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
