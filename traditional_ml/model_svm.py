import os
import pickle
from typing import List

import lightgbm as lgb
import numpy as np
from sklearn import svm

from traditional_ml.constant import train_data_path, test_data_path
from utils.constant import pkl_split_ids


class ModelSVM:
    def __init__(self):
        self.train_data = np.load(train_data_path)
        self.test_data = np.load(test_data_path)

    def start_training_svm_cls(self):
        x_train = self.train_data[:, :-1]
        y_train = self.train_data[:, -1]
        svm_clf = svm.SVC(probability=True)
        svm_clf.fit(X=x_train, y=y_train)
        return svm_clf

    def choose_threshold(self, test_split_ids: List[int], label: List[float], y_prob: List[float]):

        pass

    def start_test(self, model: svm.SVC):
        assert os.path.exists(pkl_split_ids), 'Split ids is not exist'
        test_split_ids = pickle.load(open(pkl_split_ids, 'wb'))['test']
        x_test = self.test_data[:, :-1]
        y_test = self.test_data[:, -1]
        y_prob = model.predict_proba(x_test)
