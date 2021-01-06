import copy
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import numpy as np
from flair.tokenization import SpaceTokenizer
from nlp_gym.core_components.sampler import PrioritySampler
from nlp_gym.data_pools.base import Sample
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.base_env import BaseEnv
from nlp_gym.envs.language_modeling.observation import ObservationFeaturizer, Observation
from nlp_gym.envs.language_modeling.featurizer import OneHotFeaturizerForTextGeneration
from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.language_modeling.reward import EntityF1Score
from rich import print
from gym import spaces

import logging
logger = logging.getLogger()


@dataclass(init=True)
class DataPoint:
    tokens: List[str]
    observation: Observation


class LanguageModelingEnv(BaseEnv):

    def __init__(self,
                 vocabulary: List[str],
                 window_size: int = 10,
                 reward_function: RewardFunction = None,
                 observation_featurizer: ObservationFeaturizer = None,
                 SOS: Optional[int] = None,
                 UNK: Optional[int] = None,
                 PAD: Optional[int] = None,
                 EOS: Optional[int] = None,
                 add_sos_eos: bool = True,
                 return_obs_as_vector: bool = True,
                 priority_scale: float = 0.0
                 ):
        self.sampler_for_replaying = PrioritySampler(priority_scale=priority_scale)
        self.current_datapoint: DataPoint = None
        self.start_index: int = None
        self.time_step: int = None

        self.window_size = window_size

        self.SOS = SOS if SOS is not None else vocabulary.index('SOS')
        self.UNK = UNK if UNK is not None else vocabulary.index('UNK')
        self.PAD = PAD if PAD is not None else vocabulary.index('PAD')
        self.EOS = EOS if EOS is not None else vocabulary.index('EOS')
        self.add_sos_eos = add_sos_eos

        self.action_space = self._get_action_space(vocabulary)

        reward_function = EntityF1Score(dense=True, average='micro') if reward_function is None else reward_function

        if return_obs_as_vector:
            observation_featurizer = OneHotFeaturizerForTextGeneration(self.action_space, window_size=window_size) if observation_featurizer is None else observation_featurizer
        else:
            observation_featurizer = None
        super().__init__(max_steps=None, reward_function=reward_function, observation_featurizer=observation_featurizer, return_obs_as_vector=return_obs_as_vector)


    def _tokenize(self, text: str) -> List[str]:
        return [char if char in self.action_space.actions else self.UNK for char in text]

    @staticmethod
    def _get_action_space(vocabulary: List[str]) -> ActionSpace:
        actions = copy.deepcopy(vocabulary)
        action_space = ActionSpace(actions)
        return action_space

    def reset(self, sample: Sample = None) -> Union[Observation, np.array]:
        if sample is None:
            sample = self.sampler_for_replaying.sample(size=1)[0]
        original_current_sample: Sample = sample
        logger.debug(f"New sample: '{sample.input_text}'")

        full_sample_tokens: List[str] = self._tokenize(original_current_sample.input_text)
        if self.add_sos_eos:
            full_sample_tokens = ['SOS'] + full_sample_tokens + ['EOS']
        self.start_index: int = np.random.randint(low=1, high=len(full_sample_tokens))
        input_tokens = full_sample_tokens[:self.start_index]

        self.time_step: int = self.start_index + 1
        logger.debug(f"Sample tokens: {' '.join(full_sample_tokens)}\n"
                     f"Index/Time step: {self.time_step}\n"
                     f"Input tokens: {' '.join(input_tokens)}")


        # init the featurizer with the text
        if self.observation_featurizer is not None:
            logger.debug('Calling observation_featurizer.init_on_reset')
            self.observation_featurizer.init_on_reset()

        # get initial observation
        logger.debug(f"Building observation")
        observation = Observation.build(input_tokens=input_tokens,
                                        action_history=[],
                                        observation_featurizer=self.observation_featurizer,
                                        featurize=self.return_obs_as_vector)

        # construct current data point
        logger.debug(f"Building datapoint")
        self.current_datapoint = DataPoint(tokens=full_sample_tokens,
                                           observation=observation)

        # return observation
        observation_to_return = self.current_datapoint.observation.get_vector().numpy() if self.return_obs_as_vector \
            else self.current_datapoint.observation
        return observation_to_return

    def step(self, action: int) -> Tuple[Union[Observation, np.array], float, bool, dict]:
        """
        Takes a step with the given action and returns next observation
        Returns:
            Tuple[Observation, int, bool]: observation, reward, done
        """
        # current action
        action_str: str = self.action_space.ix_to_action(action)
        targets = self.current_datapoint.tokens[self.start_index:]
        logger.debug(f"Action {action} to string {action_str}\n"
                     f"Target {targets}")

        # compute reward function
        step_reward = self.reward_function(self.current_datapoint.observation,
                                           action_str,
                                           targets)
        logger.debug(f"Step reward {step_reward}")

        # increment the time step
        self.time_step += 1
        logger.debug(f"New time step {self.time_step}")

        done = self.time_step >= len(self.current_datapoint.tokens)

        # get the updated observation
        if not done:
            logger.debug(f"Not done, building new observation")
            updated_observation = self.current_datapoint.observation.get_updated_observation(
                self.current_datapoint.tokens[:self.time_step],
                action_str,
                self.observation_featurizer,
                self.return_obs_as_vector)

            # update the current sample (just the observation)
            self.current_datapoint.observation = updated_observation
        else:
            logger.debug(f"Done! Updating action history")
            self.current_datapoint.observation.current_action_history.append(action_str)

        # return observation, reward, done, info
        observation_to_return = self.current_datapoint.observation.get_vector().numpy() if self.return_obs_as_vector \
            else self.current_datapoint.observation
        return observation_to_return, step_reward, done, {}

    def close(self):
        pass

    def render(self):
        """
        Renders the current state of the environment
        """
        print(f"[italic yellow]Step {self.time_step}[/italic yellow]")
        input_text: List[str] = ''.join(self.current_datapoint.observation.get_current_input())
        input_text = input_text.replace('SOS', '')
        actions: List[str] = [""] if self.time_step == 0 else self.current_datapoint.observation.get_current_action_history()
        action_text = ''.join(actions)
        print(f"[italic red]Input[/italic red]:{input_text} -> {action_text}")

    def add_sample(self, sample: Sample, weight: float = 1.0):
        self.sampler_for_replaying.add(sample, weight)

    def get_samples(self) -> List[Sample]:
        return self.sampler_for_replaying.get_all_samples()

    def _set_spaces(self, observation_featurizer: ObservationFeaturizer):
        low = np.full(shape=(self.window_size, observation_featurizer.get_observation_dim(),),
                      fill_value=-float('inf'),
                      dtype=np.float32)
        high = np.full(shape=(self.window_size, observation_featurizer.get_observation_dim(),),
                       fill_value=float('inf'),
                       dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
