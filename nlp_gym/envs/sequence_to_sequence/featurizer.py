import re
import string
from typing import List, Union

import flair
import torch
from flair.data import Sentence
from flair.embeddings import (BytePairEmbeddings, DocumentPoolEmbeddings,
                              Embeddings, WordEmbeddings)
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nlp_gym.envs.common.action_space import ActionSpace
from nlp_gym.envs.sequence_to_sequence.observation import ObservationFeaturizer, Observation
from nlp_gym.envs.text_generation.featurizer import OneHotEncoding


class TextPreProcessor:
    def __init__(self, language: str):
        self.language = language

    def _remove_digits(self, text: str) -> str:
        text = re.sub(r"\d+", "", text)
        return text

    def _remove_punctuation(self, text: str) -> str:
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def process(self, text: str) -> str:
        text = text.lower()
        text = self._remove_punctuation(text)
        text = self._remove_digits(text)
        text = self._remove_stop_words(text)
        text = self._stem(text)
        return text

    def _remove_stop_words(self, text: str) -> str:
        stop_words_list = stopwords.words(self.language)
        return ' '.join([word for word in text.split() if word not in stop_words_list])

    def _stem(self, text: str) -> str:
        stemmer = SnowballStemmer(language=self.language)
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def get_id(self) -> str:
        return f"advanced_{self.language}"


class EmbeddingRegistry:
    _registry_mapping = {
        "byte_pair": {
            "cls": [BytePairEmbeddings],
            "params": ["en"]
        },
        "fasttext": {
            "cls": [WordEmbeddings],
            "params": ["en-crawl"]
        },
        "stacked": {
            "cls": [WordEmbeddings, BytePairEmbeddings],
            "params": ["en-crawl", "en"]
        }
    }

    @staticmethod
    def get_embedding(embedding_type: str) -> List[Embeddings]:
        cls_ = EmbeddingRegistry._registry_mapping[embedding_type]["cls"]
        params_ = EmbeddingRegistry._registry_mapping[embedding_type]["params"]
        embeddings = [embedding_cls(embedding_param) for embedding_cls, embedding_param in zip(cls_, params_)]
        return embeddings


class DefaultFeaturizerForSequenceToSequence(ObservationFeaturizer):
    def __init__(self, action_space: ActionSpace,
                 pre_process: bool = False,
                 device: str = "cpu"):
        self.device = device
        self._setup_device()
        self.action_space = action_space
        self._setup_action_embedding()
        self._setup_encoder()
        self._current_input_embeddings = None

    def _setup_device(self):
        flair.device = torch.device(self.device)

    def _setup_encoder(self):
        raise NotImplementedError

    def _setup_action_embedding(self):
        raise NotImplementedError

    def _encode(self, input_text) -> torch.tensor:
        raise NotImplementedError

    def init_on_reset(self, input_text: Union[List[str], str]):
        self._current_input_embeddings = self._encode(input_text)

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

    def _featurize_actions(self, actions: List[int]) -> torch.Tensor:
        raise NotImplementedError

    def _get_vocab_dim(self):
        return self.action_embedding.embedding_dim

    def _get_latent_dim(self):
        return self.latent_dim

    def get_observation_dim(self) -> int:
        return self._get_input_dim() + self._get_context_dim()

    def _get_input_dim(self):
        sent = Sentence("A random text to get the embedding dimension")
        self.doc_embeddings.embed(sent)
        dim = sent[0].embedding.shape[0]
        sent.clear_embeddings()
        return dim

    def _get_context_dim(self):
        return self.action_space.size()


class DocumentPoolFeaturizer(DefaultFeaturizerForSequenceToSequence):
    def __init__(self, action_space: ActionSpace,
                 embedding_type: str = "fasttext",
                 pre_process: bool = False,
                 device: str = "cpu"):
        self.embedding_type = embedding_type
        super(DocumentPoolFeaturizer, self).__init__(action_space, device)
        self.pre_process = pre_process
        self.text_pre_processor = TextPreProcessor(language="english")

    def _encode(self, input_text: Union[List[str], str]) -> torch.tensor:
        # pooled document embeddings
        text = self.text_pre_processor.process(input_text) if self.pre_process else input_text
        self._current_input_embeddings = self.encoder(Sentence(text))

    def _setup_action_embedding(self):
        self.action_embedding = OneHotEncoding(encoding_size=self.action_space.size(), device=self.device)

    def _setup_encoder(self):
        embeddings = EmbeddingRegistry.get_embedding(self.embedding_type)
        self.encoder = DocumentPoolEmbeddings(embeddings).to(torch.device(self.device))

    def _featurize_actions(self, actions: List[int]) -> torch.Tensor:
        actions = [self.action_embedding(torch.tensor(action).type(torch.LongTensor)) for action in actions]
        return torch.stack(actions)


if __name__ == "__main__":
    feat = DocumentPoolFeaturizer(pre_process=True)
    text = 'This is an Input Text!'

    obs = Observation.build(input_str=text)
