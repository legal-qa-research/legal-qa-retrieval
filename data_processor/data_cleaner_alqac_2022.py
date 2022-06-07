import json
import pickle
import re

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity
from data_processor.question_pool import QuestionPool


def clean_question_r1():
    ques_json = json.load(open('data/ALQAC_2022/question.json', 'r'))
    cnt_err = 0
    for ques in ques_json:
        for rel_article in ques.get('relevant_articles'):
            if not rel_article.get('article_id').isnumeric():
                cnt_err += 1
                err_article_id = rel_article.get('article_id')
                regex_obj_search = re.search(r'Điều\s(\d+)\.', err_article_id)
                update_article_id = regex_obj_search.group(1).strip()
                rel_article['article_id'] = update_article_id

    json.dump(ques_json, open('data/ALQAC_2022/question_clean_v1.json', 'w'), ensure_ascii=False)
    print(cnt_err)


def count_non_exist_article():
    question_pool: QuestionPool = pickle.load(open('pkl_file/alqac_2022_question_pool.pkl', 'rb'))
    article_pool: ArticlePool = pickle.load(open('pkl_file/alqac_2022_article_pool.pkl', 'rb'))
    cnt_err = 0
    for ques in question_pool.lis_ques:
        for rel_article in ques.relevance_articles:
            if article_pool.get_position(rel_article) is None:
                cnt_err += 1
                print(rel_article.law_id, rel_article.article_id)
    return cnt_err


def clean_question_r2():
    print(count_non_exist_article())
    ques_json = json.load(open('data/ALQAC_2022/question_clean_v1.json', 'r'))
    article_pool: ArticlePool = pickle.load(open('pkl_file/alqac_2022_article_pool.pkl', 'rb'))
    new_ques_json = []
    for ques in ques_json:
        valid = True
        for rel_article in ques.get('relevant_articles'):
            aid = ArticleIdentity(rel_article)
            if article_pool.get_position(aid) is None:
                valid = False
        if valid:
            new_ques_json.append(ques)
    print('Old length: ', len(ques_json))
    print('New length: ', len(new_ques_json))
    json.dump(new_ques_json, open('data/ALQAC_2022/question_clean_v2.json', 'w'), ensure_ascii=False)


if __name__ == '__main__':
    clean_question_r2()
    pass
