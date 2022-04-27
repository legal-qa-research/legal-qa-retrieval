import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from data_processor.article_pool import ArticlePool
from utils.constant import pkl_article_pool, pkl_tfidf
from utils.utilities import get_raw_from_preproc


def preprocess_text(text: str):
    return text.lower().strip()


if __name__ == '__main__':
    stop_words = open('data/vietnamese-stopwords-dash.txt', 'r').read().split('\n')

    article_pool: ArticlePool = pickle.load(open(pkl_article_pool, 'rb'))
    corpus = [get_raw_from_preproc(article) for article in article_pool.proc_text_pool]
    vectorizer = TfidfVectorizer(stop_words=stop_words).fit(corpus)
    pickle.dump(vectorizer, open(pkl_tfidf, 'wb'))
