import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '-idp',
    "--is_dev_phase",
    default=0,
    required=False,
    type=int,
    help="Is in development phase or not",
)

parser.add_argument(
    '-si',
    "--split_ids",
    default=None,
    required=False,
    type=str,
    help="Split ids cached file",
)

parser.add_argument(
    '-itk',
    "--infer_top_k",
    default=2,
    required=False,
    type=int,
    help="Top-k when inference",
)

parser.add_argument(
    '-it',
    "--infer_threshold",
    default=0,
    required=False,
    type=float,
    help="Threshold when inference",
)

args = parser.parse_args()
