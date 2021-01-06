from dataclasses import dataclass
from typing import List, Union
from nlp_gym.envs.common.observation import BaseObservation, BaseObservationFeaturizer
import torch
import copy


@dataclass(init=True)
class Observation(BaseObservation):
    latent_vector: torch.Tensor
    current_input_index: int
    current_action_history: List[str]
    max_length: int = 10
    current_vector: torch.Tensor = None

    @classmethod
    def build(cls,
              input_index: int,
              latent_vector: torch.Tensor,
              max_length: int,
              action_history: List[str],
              observation_featurizer: BaseObservationFeaturizer,
              featurize: bool = True) -> 'Observation':
        observation = cls(latent_vector=latent_vector,
                          current_input_index=input_index,
                          max_length=max_length,
                          current_action_history=action_history)
        if featurize:
            observation.current_vector = observation_featurizer.featurize(observation)  # [B, T, D + L]
            assert observation.get_vector().shape[1] == observation_featurizer.get_observation_dim()
        return observation

    def get_updated_observation(self,
                                action: str,
                                input_index: int,
                                observation_featurizer: BaseObservationFeaturizer,
                                featurize: bool) -> 'Observation':
        updated_action_history = copy.deepcopy(self.current_action_history)
        updated_action_history.append(action)
        updated_observation = Observation.build(input_index=input_index,
                                                latent_vector=self.latent_vector,
                                                max_length=self.max_length,
                                                action_history=updated_action_history,
                                                observation_featurizer=observation_featurizer,
                                                featurize=featurize)
        return updated_observation

    def get_latent_vector(self):
        return self.latent_vector

    def get_vector(self):
        return self.current_vector

    def get_current_index(self):
        return self.current_input_index

    def get_current_action_history(self) -> List[str]:
        return self.current_action_history + ['PAD'] * (self.max_length - len(self.current_action_history))
