import json
import pickle
from typing import List

from tqdm import tqdm

from data_processor.entities.question import Question
from data_processor.preprocessor import Preprocessor


class QuestionPool:
    lis_ques: List[Question]
    proc_ques_pool: List[List[List[str]]]

    def __init__(self, ques_json_path: str):
        lis_ques_json = json.load(open(ques_json_path, 'r')).get('items')
        self.lis_ques = [Question(ques_json) for ques_json in lis_ques_json]
        self.proc_ques_pool = []

    def run_preprocess(self, preprocessor):
        self.proc_ques_pool = [preprocessor.preprocess(ques.question)
                               for ques in tqdm(self.lis_ques, desc='Preprocess Question')]


if __name__ == '__main__':
    # qp = QuestionPool('data/mini_train_ques_ans.json')
    # preproc = Preprocessor()
    # qp.run_preprocess(preproc)
    # pickle.dump(qp, open('data_processor/mini_question_pool.pkl', 'wb'))

    qp = QuestionPool('data/train_question_answer.json')
    preproc = Preprocessor()
    qp.run_preprocess(preproc)
    pickle.dump(qp, open('data_processor/question_pool.pkl', 'wb'))
    # qp = pickle.load(open('data_processor/question_pool.pkl', 'rb'))
    pass
