import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--corpus_path",
    default='legal_mlm_bert/medium_bert_corpus.txt',
    type=str,
    help="Path to corpus file",
)

parser.add_argument(
    "--model_name",
    default='vinai/phobert-base',
    type=str,
    help="BERT Model name",
)

parser.add_argument(
    "--tokenizer_name",
    default='vinai/phobert-base',
    type=str,
    help="Tokenizer for model name",
)

parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="BERT max sequence length",
)

args = parser.parse_args()
