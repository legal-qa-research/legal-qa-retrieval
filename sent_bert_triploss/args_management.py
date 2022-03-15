import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    default="haisongzhang/roberta-tiny-cased",
    type=str,
    help="name of pretrained language models",
)

parser.add_argument(
    "--max_seq_len",
    default=256,
    type=int,
    help="Max sequence length of BERT model",
)

parser.add_argument(
    "--training_file",
    default="data/used_data/train_edata.csv",
    type=str,
    help="path of training file",
)

parser.add_argument(
    "--dev_file",
    default='data/used_data/dev_edata.csv',
    type=str,
    help="path of dev file",
)

parser.add_argument(
    "--test_file",
    default='data/used_data/test_edata.csv',
    type=str,
    help="path of test file",
)

parser.add_argument(
    "--batch_size",
    default=64,
    type=int,
    help="num examples per batch",
)

parser.add_argument(
    "--lr",
    default=3e-5,
    type=float,
    help="learning rate",
)

parser.add_argument(
    "--n_epochs",
    default=20,
    type=int,
    help="num epochs required for training",
)

parser.add_argument(
    "--seed",
    default=96,
    type=int,
    help="seed for reproceduce",
)

parser.add_argument(
    "--accu_step",
    default=1,
    type=int,
    help="accu_grad_step",
)

parser.add_argument(
    "--use_aug",
    default=False,
    type=str,
    help="use data augmentation or not",
)

parser.add_argument(
    "--lang",
    default='vi',
    choices=['en', 'vi'],
    help="language of training data",
)

parser.add_argument(
    "--threshold",
    default=0.5,
    type=float,
    help="Threshold of prediction",
)

parser.add_argument(
    "--chkpoint",
    default=None,
    type=str,
    help="Checkpoint path",
)

args = parser.parse_args()
