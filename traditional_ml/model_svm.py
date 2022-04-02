import os
import pickle

import numpy as np
from sklearn import svm

from traditional_ml.constant import train_data_path, test_data_path
from traditional_ml.evaluation import Evaluation
from utils.constant import pkl_split_ids


class ModelSVM:
    def __init__(self):
        self.train_data = np.load(train_data_path)
        self.test_data = np.load(test_data_path)
        self.evaluation_machine = Evaluation()

    def start_training_svm_cls(self):
        x_train = self.train_data[:, :-1]
        y_train = self.train_data[:, -1]
        print('Training SVM .... ')
        svm_clf = svm.SVC(probability=True)
        svm_clf.fit(X=x_train, y=y_train)
        pickle.dump(svm_clf, open('traditional_ml/model_pool/svm_model.pkl', 'wb'))
        print('Training Complete')
        return svm_clf

    def start_test(self, model: svm.SVC):
        assert os.path.exists(pkl_split_ids), 'Split ids is not exist'
        x_test = self.test_data[:, :-1]
        y_prob = model.predict_proba(x_test)
        print('Evaluating Model...')
        self.evaluation_machine.start_eval(0, 0, y_prob)
        print('Evaluating Complete')

    def run(self):
        model = self.start_training_svm_cls()
        self.start_test(model)


if __name__ == '__main__':
    from data_processor.article_pool import ArticlePool

    ms = ModelSVM()
    ms.run()
