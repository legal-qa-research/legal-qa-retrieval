import json
from typing import List, Dict

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
    def __init__(self, ground_truth_json_path: str, predict_json_path: str):
        self.ground_truth = json.load(open(ground_truth_json_path, 'r'))
        self.predict = json.load(open(predict_json_path, 'r'))

    def start_evaluate(self):
        f2score = 0
        precision = 0
        recall = 0
        for g_ques in self.ground_truth:
            qid = g_ques.get('question_id')
            p_ques = [q for q in self.predict if q.get('question_id') == qid]
            assert len(p_ques) == 1, 'Duplicate or not found question'
            p_ques = p_ques[0]
            f2i, precision_i, recall_i = cal_single_f2score(ground_truth_relevant=g_ques.get('relevant_articles'),
                                                            predict_relevant=p_ques.get('relevant_articles'))
            f2score += f2i
            precision += precision_i
            recall += recall_i
        print('Recall: ', recall / len(self.ground_truth))
        print('Precision: ', precision / len(self.ground_truth))
        print(f2score / len(self.ground_truth))


if __name__ == '__main__':
    se = SubmissionEvaluator(ground_truth_json_path='data/kse_private_test_question.json',
                             predict_json_path='utils/submission/top_k_infer_kse_v5_5ep.json')
    se.start_evaluate()
