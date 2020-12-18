from nlp_gym.envs.common.reward import RewardFunction
from nlp_gym.envs.text_generation.observation import Observation
from typing import List
import copy


class CounterScore(RewardFunction):

    def __init__(self, string_to_count: str = 'ab'):
        self.string_to_count = string_to_count

    def __call__(self, observation: Observation, action: str, *args, **kwargs) -> float:
        current_action_history = copy.deepcopy(observation.get_current_action_history())
        current_action_history.append(action)
        reward = self.reward_fn(current_action_history)
        return reward

    def reward_fn(self, targets: List[str]):
        targets = ''.join(targets)
        return targets.count(self.string_to_count) / len(targets)
