import argparse


class RefSupArgument:
    model_name: str
    tokenizer_name: str
    embed_size: int
    is_dev_phase: int
    is_narrow_article: int
    max_seq_len: int
    evaluation_steps: int
    batch_size: int
    lr: float
    n_epochs: int
    scheduler: str
    infer_threshold: float
    infer_top_k: int
    split_ids: str
    use_segmenter: int
    root_dir: str
    n_gpus: int
    resume_checkpoint: str
    pretrained_sent_bert: str


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    default="vinai/phobert-base",
    type=str,
    help="name of pretrained language models",
)

parser.add_argument(
    "--tokenizer_name",
    default='vinai/phobert-base',
    type=str,
    help="BERT Tokenizer name",
)

parser.add_argument(
    "--pretrained_sent_bert",
    default=None,
    type=str,
    help="Path of pretrained sent bert",
)

parser.add_argument(
    "--is_dev_phase",
    default=0,
    type=int,
    help="Is in development phase or not",
)

parser.add_argument(
    "--is_narrow_article",
    default=0,
    type=int,
    help="Narrow down article content according task 2 answer",
)

parser.add_argument(
    "--max_seq_len",
    default=256,
    type=int,
    help="Max sequence length of BERT model",
)

parser.add_argument(
    "--evaluation_steps",
    default=4000,
    type=int,
    help="Number of step training for each evaluation",
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
    "--scheduler",
    default='warmupcosine',
    type=str,
    help="Type of scheduler for sentbert",
)

parser.add_argument(
    "--infer_threshold",
    default=0.5,
    type=float,
    help="Threshold of prediction",
)

parser.add_argument(
    "--split_ids",
    default=None,
    type=str,
    help="Split ids cached file",
)

parser.add_argument(
    "--infer_top_k",
    default=2,
    type=int,
    help="Top-k when inference",
)

parser.add_argument(
    "--use_segmenter",
    default=1,
    type=int,
    help="Use segmenter or not",
)

parser.add_argument(
    "--root_dir",
    default='./',
    type=str,
    help="Root dir for save checkpoint",
)

parser.add_argument(
    "--n_gpus",
    default=3,
    type=int,
    help="Number of GPUS for training",
)

parser.add_argument(
    "--resume_checkpoint",
    default=None,
    type=str,
    help="Reusume training from checkpoint",
)

parser.add_argument(
    "--embed_size",
    default=768,
    type=int,
    help="Embedding size of sentence",
)

args = parser.parse_args(namespace=RefSupArgument())
pass
