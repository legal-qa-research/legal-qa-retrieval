import numpy as np

from data_processor.article_pool import ArticlePool
from traditional_ml.evaluation import Evaluation
from traditional_ml.raw_input_example import RawInputExample

if __name__ == '__main__':
    evaluation_machine = Evaluation()
    y_prob = np.load('traditional_ml/prob_test_set/result_test_set.npy')
    evaluation_machine.start_eval(1, 1, y_prob)
