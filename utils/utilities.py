from random import shuffle


def get_raw_from_preproc(preproc):
    return ' '.join([' '.join(sent) for sent in preproc])


def split_ids(n_samples: int, test_size=0.2):
    train_size = 1 - test_size
    cut_pos = int(n_samples * train_size)

    lis_id = [i for i in range(n_samples)]
    shuffle(lis_id)
    return lis_id[:cut_pos], lis_id[cut_pos:]
