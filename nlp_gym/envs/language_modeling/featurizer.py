from typing import List, Union, Optional

import torch
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.common.observation import BaseObservationFeaturizer
from nlp_gym.envs.language_modeling.observation import Observation


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

    def __init__(self, action_space: ActionSpace, window_size: int = 10, device: str = "cpu"):
        self.window_size = window_size
        self.device = torch.device(device)

        # tokens are the strings, actions are the corresponding ints
        self.action_embedding = OneHotEncoding(encoding_size=action_space.size(), device=self.device)
        self.action_space = action_space
        self._current_token_embeddings: Optional[List[torch.tensor]] = None

    def init_on_reset(self, **kwargs):
        pass

    def featurize(self, observation: Observation) -> torch.Tensor:
        tokens = observation.get_current_input()[-self.window_size:]
        tokens += ['PAD']*(self.window_size - len(tokens))
        action_indices = [self.action_space.action_to_ix(token) for token in tokens]
        return self._featurize_actions(action_indices)

    def get_observation_dim(self) -> int:
        return self._get_vocab_dim()

    def _featurize_actions(self, actions: List[int]) -> torch.Tensor:
        actions = [self.action_embedding(torch.tensor(action).type(torch.LongTensor)) for action in actions]
        return torch.stack(actions)

    def _get_vocab_dim(self):
        return self.action_embedding.embedding_dim


def main():
    vocab = ['SOS', 'EOS', 'PAD', 'UNK'] + list('abcdefg')
    print('VOCAB SIZE:', len(vocab))
    action_space = ActionSpace(actions=vocab)
    device = 'cpu'

    text = ['SOS'] + list('abcdefgabcdefg') + ['EOS']
    start_index = 5

    featurizer = OneHotFeaturizerForTextGeneration(action_space, window_size=10, device='cpu')
    observation = Observation(current_input_tokens=text[:start_index],
                              current_action_history=[],
                              current_vector=None)

    output = featurizer.featurize(observation)
    print(output.shape)
    print(output)


if __name__ == '__main__':
    main()

