import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--is_dev_phase",
    default=0,
    type=int,
    help="Is in development phase or not",
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
    "--infer_threshold",
    default=0,
    type=float,
    help="Threshold when inference",
)

args = parser.parse_args()
