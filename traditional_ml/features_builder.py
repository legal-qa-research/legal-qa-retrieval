import pickle
from typing import List

import fasttext
import numpy as np
from fasttext.FastText import _FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from traditional_ml.args_management import args
from traditional_ml.constant import train_examples_path, test_examples_path
from traditional_ml.data import Data
from traditional_ml.raw_input_example import RawInputExample
from utils.constant import pkl_question_pool, pkl_article_pool, pkl_cached_rel, pkl_split_ids, pkl_tfidf
from utils.utilities import calculate_jaccard_sim
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class FeaturesBuilder:
    def __init__(self, args, fasttext_model_path):
        self.args = args
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.train_phase_data = Data(pkl_question_pool_path=pkl_question_pool, pkl_article_pool_path=pkl_article_pool,
                                     pkl_cached_rel_path=pkl_cached_rel, pkl_cached_split_ids=pkl_split_ids,
                                     args=self.args)
        self.tfidf_vtr: TfidfVectorizer = pickle.load(open(pkl_tfidf, 'rb'))

    @staticmethod
    def cal_avg_word_embed_vec(ft: _FastText, tfidf_vtr: TfidfVectorizer, tfidf_matrix: csr_matrix):
        v = np.stack(
            [ft.get_word_vector(tfidf_vtr.get_feature_names_out()[wid]) * tfidf_matrix[0, wid]
             for wid in tfidf_matrix.nonzero()[1]])
        return v.mean(axis=0)

    @staticmethod
    def cal_feature_from_inp(input_exp: RawInputExample, ft: _FastText, tfidf_vtr: TfidfVectorizer,
                             tfidf_matrix: csr_matrix):
        ques_art_tfidf = tfidf_vtr.transform([' '.join(input_exp.ques), ' '.join(input_exp.articles)])
        cosine = cosine_similarity(ques_art_tfidf)
        ques_vec = FeaturesBuilder.cal_avg_word_embed_vec(ft=ft, tfidf_matrix=ques_art_tfidf[0], tfidf_vtr=tfidf_vtr)
        art_vec = FeaturesBuilder.cal_avg_word_embed_vec(ft=ft, tfidf_matrix=ques_art_tfidf[1], tfidf_vtr=tfidf_vtr)

        jaccard_score = calculate_jaccard_sim(u=input_exp.ques, v=input_exp.articles)
        label = input_exp.label
        return np.concatenate((ques_vec, art_vec, [cosine[0, 1], jaccard_score, label]), axis=0)

    def build_vec(self, ft: _FastText, tfidf_matrix: csr_matrix) -> np.ndarray:
        v = np.stack(
            [ft.get_word_vector(self.tfidf_vtr.get_feature_names_out()[wid]) * tfidf_matrix[0, wid]
             for wid in tfidf_matrix.nonzero()[1]])
        # v = np.stack([ft.get_word_vector(tok) for tok in lis_tok])
        return v.mean(axis=0)

    def build_feature(self, lis_examples: List[RawInputExample]) -> np.ndarray:
        data = []
        for input_exp in tqdm(lis_examples, desc='Build Features'):
            ques_art_tfidf = self.tfidf_vtr.transform([' '.join(input_exp.ques), ' '.join(input_exp.articles)])
            cosine = cosine_similarity(ques_art_tfidf)
            ques_vec = self.build_vec(ft=self.fasttext_model, tfidf_matrix=ques_art_tfidf[0])
            art_vec = self.build_vec(ft=self.fasttext_model, tfidf_matrix=ques_art_tfidf[1])

            jaccard_score = calculate_jaccard_sim(u=input_exp.ques, v=input_exp.articles)
            label = input_exp.label
            data.append(np.concatenate((ques_vec, art_vec, [cosine[0, 1], jaccard_score, label]), axis=0))
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
    a = np.load('traditional_ml/feature_data/train_data_vnlaw_kse_jac_tfidf.npy')
    b = np.load('traditional_ml/feature_data/test_data_vnlaw_kse_jac_tfidf.npy')
    print(a.shape)
    print(b.shape)
    pass


if __name__ == '__main__':
    from data_processor.article_pool import ArticlePool
    from data_processor.question_pool import QuestionPool
    from bm25_ranking.bm25_ranker import Bm25Ranker
    from bm25_ranking.bm25_ranker_cached import Bm25RankerCached

    fb = FeaturesBuilder(args=args, fasttext_model_path='traditional_ml/pretrained_fasttext/vnlaw_ft.bin')
    # fb = FeaturesBuilder(args=args, fasttext_model_path='traditional_ml/pretrained_fasttext/wiki.vi.bin')
    fb.start_build(fast_text_type='vnlaw_kse_jac_tfidf')

    test_built_feature()
