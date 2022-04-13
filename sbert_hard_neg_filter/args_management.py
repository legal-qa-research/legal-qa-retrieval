import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name_or_path",
    default="vinai/phobert-base",
    type=str,
    help="name of pretrained language models",
)

parser.add_argument(
    "--split_ids",
    default=None,
    type=str,
    help="Split ids cached file",
)

parser.add_argument(
    "--is_dev_phase",
    default=0,
    type=int,
    help="Dev phase or not",
)

args = parser.parse_args()
