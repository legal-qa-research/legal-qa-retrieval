import json
import os.path
import pickle
from typing import List

from tqdm import tqdm

from data_processor.entities.question import Question
from data_processor.preprocessor import Preprocessor
from utils.constant import json_train_question, pkl_question_pool


class QuestionPool:
    lis_ques: List[Question]
    proc_ques_pool: List[List[List[str]]]

    def __init__(self, ques_json_path: str = None):
        if ques_json_path is not None and os.path.exists(ques_json_path):
            lis_ques_json = json.load(open(ques_json_path, 'r'))
            self.lis_ques: List[Question] = [Question(ques_json) for ques_json in lis_ques_json]
            self.proc_ques_pool = []
            self.proc_answer_pool = []

    def extract_sub_set(self, lis_ids: List[int]):
        new_lis_ques = [self.lis_ques[i] for i in lis_ids]
        new_proc_ques_pool = [self.proc_ques_pool[i] for i in lis_ids]
        new_ques_pool = QuestionPool()
        new_ques_pool.lis_ques = new_lis_ques
        new_ques_pool.proc_ques_pool = new_proc_ques_pool
        return new_ques_pool

    def run_preprocess(self, preprocessor):
        self.proc_ques_pool = [preprocessor.preprocess(ques.question) if ques.question is not None else ''
                               for ques in tqdm(self.lis_ques, desc='Preprocess Question')]
        self.proc_answer_pool = None
        self.proc_answer_pool = [preprocessor.preprocess(ques.answer) if ques.answer is not None else ''
                                 for ques in tqdm(self.lis_ques, desc='Preprocess answer')]


if __name__ == '__main__':
    # qp = QuestionPool('data/mini_train_ques_ans.json')
    # preproc = Preprocessor()
    # qp.run_preprocess(preproc)
    # pickle.dump(qp, open('data_processor/mini_question_pool.pkl', 'wb'))
    #
    # qp = QuestionPool('data/ALQAC_2022/ALQAC_test_release.json')
    # preproc = Preprocessor()
    # qp.run_preprocess(preproc)
    # pickle.dump(qp, open('pkl_file/alqac_2022_test_question_pool.pkl', 'wb'))
    #
    # qp = pickle.load(open('pkl_file/alqac_2022_test_question_pool.pkl', 'rb'))
    #
    qp = QuestionPool(json_train_question)
    preproc = Preprocessor()
    qp.run_preprocess(preproc)
    pickle.dump(qp, open(pkl_question_pool, 'wb'))
    qp = pickle.load(open(pkl_question_pool, 'rb'))
    pass
