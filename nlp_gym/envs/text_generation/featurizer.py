from typing import List, Union, Optional

import torch
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.observation import BaseObservationFeaturizer
from nlp_gym.envs.text_generation.observation import Observation


class OneHotEncoding(object):
    def __init__(self, encoding_size: int, device: torch.device = torch.device('cpu')):
        self.encoding_size = encoding_size
        self.device = device

    def __call__(self, indexes: torch.LongTensor) -> torch.FloatTensor:
        one_hot = torch.nn.functional.one_hot(indexes, self.encoding_size).to(self.device)
        return one_hot.float()

    @property
    def embedding_dim(self):
        return self.encoding_size


class OneHotFeaturizerForTextGeneration(BaseObservationFeaturizer):

    def __init__(self, action_space: ActionSpace, latent_dim: int, device: str = "cpu"):
        self.device = torch.device(device)

        self.latent_dim = latent_dim

        self.action_embedding = OneHotEncoding(encoding_size=action_space.size(), device=self.device)
        self.action_space = action_space
        self._current_token_embeddings: Optional[List[torch.tensor]] = None

    def init_on_reset(self, **kwargs):
        pass

    def featurize(self, observation: Observation) -> torch.Tensor:
        if observation.current_vector is None:
            base_zeros = torch.zeros(observation.max_length, self._get_vocab_dim())
            latent_vectors = observation.latent_vector.view(1, -1).repeat(observation.max_length, 1)
            observation.current_vector = torch.cat((base_zeros, latent_vectors), dim=1)
        action_indices = [self.action_space.action_to_ix(action) for action in observation.get_current_action_history()]
        vocab_embeddings = self._featurize_actions(action_indices)
        new_vector = observation.current_vector
        new_vector[:len(observation.get_current_action_history()), :self._get_vocab_dim()] = vocab_embeddings
        return new_vector

    def get_observation_dim(self) -> int:
        return self._get_vocab_dim() + self._get_latent_dim()

    def _featurize_actions(self, actions: List[int]) -> torch.Tensor:
        actions = [self.action_embedding(torch.tensor(action).type(torch.LongTensor)) for action in actions]
        return torch.stack(actions)

    def _get_vocab_dim(self):
        return self.action_embedding.embedding_dim

    def _get_latent_dim(self):
        return self.latent_dim


if __name__ == '__main__':
    vocab = ['SOS', 'EOS'] + list('abcdefg')
    action_space = ActionSpace(actions=vocab)
    latent_dim = 10
    device = 'cpu'

    featurizer = OneHotFeaturizerForTextGeneration(action_space, latent_dim, device)
    observation = Observation(latent_vector=torch.rand(latent_dim),
                              current_input_index=0,
                              current_action_history=['SOS', 'a', 'b'],
                              max_length=10,
                              current_vector=None)

    output = featurizer.featurize(observation)
    print(output.shape)
