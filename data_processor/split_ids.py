import pickle

from data_processor.question_pool import QuestionPool
from utils.constant import pkl_question_pool
from random import shuffle


def start_split_and_save(dev_size=0.1, test_size=0.1):
    question_pool: QuestionPool = pickle.load(open(pkl_question_pool, 'rb'))
    lis_ids = [i for i in range(len(question_pool.lis_ques))]
    shuffle(lis_ids)
    dev_len = int(len(lis_ids) * dev_size)
    test_len = int(len(lis_ids) * test_size)
    train_len = len(lis_ids) - dev_len - test_len
    save_obj = {
        'train': lis_ids[:train_len],
        'dev': lis_ids[train_len:train_len + dev_len],
        'test': lis_ids[train_len + dev_len:train_len + dev_len + test_len]
    }

    pickle.dump(save_obj, open('pkl_file/alqac_2022_split_ids.pkl', 'wb'))


if __name__ == '__main__':
    start_split_and_save()
