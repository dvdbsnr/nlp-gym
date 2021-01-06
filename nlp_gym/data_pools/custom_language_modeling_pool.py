from nlp_gym.data_pools.language_modeling_pool import LanguageModelingPool
from nlp_gym.data_pools.base import Sample
from nlp_gym.util import get_sample_weights
from nlp_gym import aapd_data_path
from collections import defaultdict
from typing import List, Dict
from nltk.corpus import reuters
import os
import random


class AAPDDataPool(LanguageModelingPool):
    """
    Source repo: https://git.uwaterloo.ca/jimmylin/Castor-data/tree/master/datasets/AAPD
    Dataset for paper: https://arxiv.org/pdf/1806.04822.pdf
    """

    @classmethod
    def prepare(cls, split: str):
        documents = []
        vocab = set()
        with open(os.path.join(aapd_data_path, f"text_{split}")) as text_file:
            for text in text_file:
                if text != "":
                    text = text.strip()[:20]
                    chars = set([char for char in text])
                    vocab.update(chars)
                    documents.append(Sample(input_text=text, oracle_label=None))
        random.shuffle(documents)
        weights = [1. / len(documents)] * len(documents)
        vocab: List[str] = list(vocab)
        return cls(documents, vocab, weights)


if __name__ == "__main__":
    pool = AAPDDataPool.prepare(split="train")
    print(len(pool))
    print(pool[2])
