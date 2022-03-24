from typing import List, Tuple

from tqdm import tqdm

from data_processor.entities.article_identity import ArticleIdentity
from data_processor.entities.legal_corpus import LegalCorpus
from data_processor.preprocessor import Preprocessor
import pickle


class ArticlePool:
    def __init__(self, legal_corpus: LegalCorpus):
        self.legal_corpus = legal_corpus
        self.text_pool = [article.text for law in self.legal_corpus.laws for article in law.articles]
        self.proc_text_pool = None
        self.title_pool = [article.title for law in self.legal_corpus.laws for article in law.articles]
        self.proc_title_pool = None
        self.article_identity: List[ArticleIdentity] = [
            ArticleIdentity({'law_id': law.law_id, 'article_id': article.article_id})
            for law in self.legal_corpus.laws for article in law.articles]

    def run_preprocess(self, preprocessor: Preprocessor):
        self.proc_text_pool = [preprocessor.preprocess(_txt) for _txt in
                               tqdm(self.text_pool, desc='Preprocess text')]
        self.proc_title_pool = [preprocessor.preprocess(_txt) for _txt in
                                tqdm(self.title_pool, desc='Preprocess title')]

    def get_position(self, article_identity: ArticleIdentity):
        for i, identity in enumerate(self.article_identity):
            if identity[0] == article_identity.law_id and identity[1] == article_identity.article_id:
                return i
        return None


if __name__ == '__main__':
    # lc = LegalCorpus(json_path='data/mini_legal_corpus.json')
    # article_pool = ArticlePool(lc)
    # preproc = Preprocessor()
    # article_pool.run_preprocess(preproc)
    # pickle.dump(article_pool, open('data_processor/mini_article_pool.pkl', 'wb'))
    # article_pool = pickle.load(open('data_processor/mini_article_pool.pkl', 'rb'))

    lc = LegalCorpus(json_path='data/legal_corpus.json')
    article_pool = ArticlePool(lc)
    preproc = Preprocessor()
    article_pool.run_preprocess(preproc)
    pickle.dump(article_pool, open('data_processor/article_pool.pkl', 'wb'))
    # article_pool = pickle.load(open('data_processor/article_pool.pkl', 'rb'))
