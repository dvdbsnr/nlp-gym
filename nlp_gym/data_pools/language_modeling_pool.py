from nlp_gym.data_pools.base import DataPool, Sample
from typing import Tuple, List
from abc import abstractmethod
import random


class LanguageModelingPool(DataPool):
    def __init__(self, samples: List[Sample],  vocabulary: List[str], sample_weights: List[float]):
        self._samples, self._vocabulary, self._weights = samples, vocabulary, sample_weights
        self.pool_length = len(self._samples)

    def __len__(self):
        return self.pool_length

    def __getitem__(self, ix: int) -> Tuple[Sample, float]:
        if ix >= self.pool_length:
            raise StopIteration
        sample = self._samples[ix]
        weight = self._weights[ix]
        return sample, weight

    def get_vocabulary(self) -> List[str]:
        return self._vocabulary

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @classmethod
    @abstractmethod
    def prepare(cls, **args) -> 'LanguageModelingPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List['LanguageModelingPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(self.pool_length * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix], self._weights[start_ix: end_ix]))
            start_ix = end_ix
        return pools
