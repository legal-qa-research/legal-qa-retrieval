from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from ref_support_bert.ref_sup_data.ref_sup_sample import RefSupSample, create_lis_sample
from ref_support_bert.args_management import args


class RefSupDataset(Dataset):
    def __init__(self, lis_sample: List[List[Tensor]]):
        self.lis_sample = lis_sample

    def __getitem__(self, index) -> List[Tensor]:
        return self.lis_sample[index]

    def __len__(self):
        return len(self.lis_sample)


def get_ref_sup_dataloader():
    ref_sup_sample = create_lis_sample()
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(model_name_or_path=args.load_chk_point, device=device)
    lis_data = []
    with torch.no_grad():
        for sample in ref_sup_sample:
            lis_text: List[str] = [sample.query_text] + [article for article in sample.lis_article]
            lis_encode_vec: List[Tensor] = encoder.encode(lis_text, convert_to_tensor=True, convert_to_numpy=False)
            lis_data.append(lis_encode_vec)
    ref_sup_dataset = RefSupDataset(lis_data)
    return DataLoader(dataset=ref_sup_dataset, batch_size=None, sampler=None)
