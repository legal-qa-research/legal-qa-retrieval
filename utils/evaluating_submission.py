import json
import pickle
from typing import List, Dict

from data_processor.article_pool import ArticlePool
from data_processor.entities.article_identity import ArticleIdentity

ESP = 1e-10


def cal_single_f2score(ground_truth_relevant: List[Dict], predict_relevant: List[Dict]):
    true_pos = 0
    for article_identity in ground_truth_relevant:
        is_contain = False
        for predict_article_identity in predict_relevant:
            if predict_article_identity.get('law_id') == article_identity.get(
                    'law_id') and predict_article_identity.get('article_id') == article_identity.get('article_id'):
                is_contain = True
                break
        if is_contain:
            true_pos += 1

    precision = true_pos / (len(predict_relevant) + ESP)
    recall = true_pos / len(ground_truth_relevant)
    return 5 * precision * recall / (4 * precision + recall + ESP), precision, recall


class SubmissionEvaluator:
    def __init__(self, ground_truth_json_path: str, predict_json_path: str, article_pool: str):
        self.ground_truth = json.load(open(ground_truth_json_path, 'r'))
        self.predict = json.load(open(predict_json_path, 'r'))
        self.article_pool: ArticlePool = pickle.load(open(article_pool, 'rb'))

    def count_ambiguous_article(self, p_ques: Dict, g_ques: Dict):
        total_art_ambi = 0
        for p_article in p_ques.get('relevant_articles'):
            is_ambi = False
            for g_article in g_ques.get('relevant_articles'):
                if p_article.get('law_id') == g_article.get('law_id') \
                        and p_article.get('article_id') != g_article.get('article_id'):
                    is_ambi = True
            if is_ambi:
                total_art_ambi += 1
        return total_art_ambi

    def start_evaluate(self):
        f2score = 0
        precision = 0
        recall = 0
        n_ques_ambi = 0
        for g_ques in self.ground_truth:
            qid = g_ques.get('question_id')
            p_ques = [q for q in self.predict if q.get('question_id') == qid]
            assert len(p_ques) == 1, 'Duplicate or not found question'
            p_ques = p_ques[0]
            f2i, precision_i, recall_i = cal_single_f2score(ground_truth_relevant=g_ques.get('relevant_articles'),
                                                            predict_relevant=p_ques.get('relevant_articles'))
            cnt_art_ambi = self.count_ambiguous_article(p_ques=p_ques, g_ques=g_ques)
            if cnt_art_ambi > 0:
                n_ques_ambi += 1

            if precision_i < 0.8 or recall_i < 1.0:
                print('=' * 100)
                print(precision_i, recall_i)
                print(g_ques.get('text'))
                print('Relevant: ')
                for rel_article in g_ques.get('relevant_articles'):
                    print(rel_article)
                    print(
                        self.article_pool.text_pool[self.article_pool.get_position(ArticleIdentity(rel_article))])
                    print('-' * 50)
                print('Predict: ')
                for rel_article in p_ques.get('relevant_articles'):
                    print(rel_article)
                    print(
                        self.article_pool.text_pool[self.article_pool.get_position(ArticleIdentity(rel_article))])
            f2score += f2i
            precision += precision_i
            recall += recall_i
        print('Recall: ', recall / len(self.ground_truth))
        print('Precision: ', precision / len(self.ground_truth))
        print('F2 Score', f2score / len(self.ground_truth))
        print('Number of question have ambiguous predict: ', n_ques_ambi / len(self.ground_truth))


if __name__ == '__main__':
    se = SubmissionEvaluator(ground_truth_json_path='data/kse_private_test_question.json',
                             predict_json_path='utils/submission/threshold_infer_kse_v9.json',
                             article_pool='pkl_file/kse_article_pool.pkl')
    se.start_evaluate()
