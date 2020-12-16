
from nlp_gym.envs.common.observation import BaseObservation
from abc import ABC, abstractmethod
from typing import List, Optional


class RewardFunction(ABC):

    @classmethod
    @abstractmethod
    def __call__(self, observation: BaseObservation, action: str, targets: Optional[List[str]]) -> float:
        """[summary]

        Args:
            observation (Observation): current observation at t
            action (str): current action at t
            targets (List[str]): targets of the current sample

        Returns:
            - a scalar reward
        """
        raise NotImplementedError
