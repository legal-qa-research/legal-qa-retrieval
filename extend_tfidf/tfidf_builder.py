import pickle
from typing import Union, List

from sklearn.feature_extraction.text import TfidfVectorizer

from data_processor.article_pool import ArticlePool
from utils.constant import pkl_article_pool


class TfidfBuilder:
    def __init__(self):
        self.corpus: Union[None, List[str]] = None
        self.vectorizer: Union[None, TfidfVectorizer] = None
        self.arti_pool_alqac_2022: Union[None, ArticlePool] = None

    def build_corpus(self):
        self.arti_pool_alqac_2022 = pickle.load(open(pkl_article_pool, 'rb'))

