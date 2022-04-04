import json
import pickle
from typing import List

import fasttext
import numpy as np
from fasttext.FastText import _FastText
from tqdm import tqdm

from traditional_ml.args_management import args
from traditional_ml.constant import train_examples_path, test_examples_path
from traditional_ml.data import Data
from traditional_ml.raw_input_example import RawInputExample
from utils.constant import pkl_question_pool, pkl_article_pool, pkl_cached_rel, pkl_split_ids
from utils.utilities import build_vec


class FeaturesBuilder:
    def __init__(self, args, fasttext_model_path):
        self.args = args
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.train_phase_data = Data(pkl_question_pool_path=pkl_question_pool, pkl_article_pool_path=pkl_article_pool,
                                     pkl_cached_rel_path=pkl_cached_rel, pkl_cached_split_ids=pkl_split_ids,
                                     args=self.args)

    def build_feature(self, lis_examples: List[RawInputExample]) -> np.ndarray:
        data = []
        for input_exp in tqdm(lis_examples, desc='Build Features'):
            ques_vec = build_vec(ft=self.fasttext_model, lis_tok=input_exp.ques)
            art_vec = build_vec(ft=self.fasttext_model, lis_tok=input_exp.articles)
            label = input_exp.label
            data.append(np.concatenate((ques_vec, art_vec, [label]), axis=0))
        return np.stack(data)

    def start_build(self, fast_text_type: str = ''):
        train_examples, test_examples = self.train_phase_data.build_dataset()
        pickle.dump(train_examples, open(train_examples_path, 'wb'))
        pickle.dump(test_examples, open(test_examples_path, 'wb'))
        train_data = self.build_feature(train_examples)
        test_data = self.build_feature(test_examples)
        np.save(f'traditional_ml/feature_data/train_data_{fast_text_type}.npy', train_data)
        np.save(f'traditional_ml/feature_data/test_data_{fast_text_type}.npy', test_data)


def test_built_feature():
    a = np.load('traditional_ml/feature_data/train_data_general.npy')
    b = np.load('traditional_ml/feature_data/test_data_general.npy')
    pass


if __name__ == '__main__':
    from data_processor.article_pool import ArticlePool
    from data_processor.question_pool import QuestionPool
    from bm25_ranking.bm25_ranker import Bm25Ranker
    from bm25_ranking.bm25_ranker_cached import Bm25RankerCached

    fb = FeaturesBuilder(args=args, fasttext_model_path='traditional_ml/pretrained_fasttext/vnlaw_ft.bin')
    # fb = FeaturesBuilder(args=args, fasttext_model_path='traditional_ml/pretrained_fasttext/wiki.vi.bin')
    fb.start_build(fast_text_type='general')

    test_built_feature()
