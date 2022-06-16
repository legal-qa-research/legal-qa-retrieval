from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import Dataset, DataLoader
from ref_support_bert.ref_sup_data.ref_sup_sample import RefSupSample, create_lis_sample
from ref_support_bert.args_management import args


class RefSupDataset(Dataset):
    def __init__(self, lis_sample: List[Tuple[PackedSequence, float]]):
        self.lis_sample = lis_sample

    def __getitem__(self, index) -> Tuple[PackedSequence, float]:
        return self.lis_sample[index]

    def __len__(self):
        return len(self.lis_sample)


def get_ref_sup_dataloader():
    ref_sup_sample = create_lis_sample()
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(model_name_or_path=args.load_chk_point, device=device)
    lis_data: List[Tuple[PackedSequence, float]] = []
    with torch.no_grad():
        for sample in ref_sup_sample:
            lis_text: List[str] = [sample.query_text] + [article for article in sample.lis_article]
            lis_encode_vec: Tensor = encoder.encode(lis_text, convert_to_tensor=True, convert_to_numpy=False)
            lis_data.append((pack_sequence([lis_encode_vec]), 1.0))
    ref_sup_dataset = RefSupDataset(lis_data)
    return DataLoader(dataset=ref_sup_dataset, batch_size=None, sampler=None)
