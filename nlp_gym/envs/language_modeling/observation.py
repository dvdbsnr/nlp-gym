from dataclasses import dataclass
from typing import List, Union, Optional
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer
from abc import abstractmethod
import torch
import copy


class ObservationFeaturizer(BaseObservationFeaturizer):

    @abstractmethod
    def init_on_reset(self, input_text: Union[List[str], str]):
        """
        Takes an input text (sentence) or list of token strings and featurizes it or prepares it
        This function would be called in env.reset()
        """
        raise NotImplementedError


@dataclass(init=True)
class Observation(BaseObservation):
    current_input_tokens: List[str]
    current_action_history: List[str]
    current_vector: Optional[torch.Tensor] = None

    def get_current_input(self):
        return self.current_input_tokens

    def get_current_action_history(self) -> List[str]:
        return self.current_action_history

    def get_vector(self) -> torch.Tensor:
        return self.current_vector

    @classmethod
    def build(cls,
              input_tokens: List[str],
              action_history: List[str],
              observation_featurizer: Optional[ObservationFeaturizer],
              featurize: bool) -> 'Observation':
        observation = Observation(input_tokens, action_history)
        if featurize:
            if observation_featurizer is None:
                raise ValueError
            observation.current_vector = observation_featurizer.featurize(observation)
            assert observation.get_vector().shape[1] == observation_featurizer.get_observation_dim()
        return observation

    def get_updated_observation(self,
                                input_tokens: List[str],
                                action: str,
                                observation_featurizer: ObservationFeaturizer,
                                featurize: bool) -> 'Observation':
        updated_action_history = copy.deepcopy(self.current_action_history)
        updated_action_history.append(action)
        updated_observation = Observation.build(input_tokens,
                                                updated_action_history,
                                                observation_featurizer,
                                                featurize)
        return updated_observation
