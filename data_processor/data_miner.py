import pickle

from data_processor.article_pool import ArticlePool
from data_processor.question_pool import QuestionPool
from utils.utilities import get_flat_list_from_preproc, get_raw_from_preproc

if __name__ == '__main__':
    # article_pool: ArticlePool = pickle.load(open('pkl_file/alqac_2022_article_pool.pkl', 'rb'))
    # for proc_text_pool in article_pool.proc_text_pool:
    #     lis_token = get_flat_list_from_preproc(proc_text_pool)
    #     if len(lis_token) >= 256:
    #         print(get_raw_from_preproc(proc_text_pool))

    ques_pool: QuestionPool = pickle.load(open('pkl_file/alqac_2022_question_pool.pkl', 'rb'))
    print(len(ques_pool.proc_ques_pool))
