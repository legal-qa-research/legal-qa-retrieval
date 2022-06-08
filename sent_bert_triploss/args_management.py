import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    # default="haisongzhang/roberta-tiny-cased",
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
    "--output_path",
    default='sent_bert_triploss/output_model',
    type=str,
    help='Path of output model',
)

parser.add_argument(
    "--chk_point",
    default='sent_bert_triploss/chkpoint',
    type=str,
    help="Checkpoint path",
)

parser.add_argument(
    "--load_chk_point",
    default=None,
    type=str,
    help="Load checkpoint path for continue training",
)

parser.add_argument(
    "--threshold",
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
    "--chk_limit",
    default=5,
    type=int,
    help="Maximum number of checkpoint to save",
)

parser.add_argument(
    "--use_contrast_loss_fn",
    default=1,
    type=int,
    help="Use contrast loss or not",
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

args = parser.parse_args()
