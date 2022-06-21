from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from ref_support_bert.ref_sup_data.ref_sup_sample import RefSupSample
from ref_support_bert.args_management import args


class RefSupDataset(Dataset):
    def __init__(self, lis_sample: List[Tuple[Tensor, float]]):
        self.lis_sample = lis_sample

    def __getitem__(self, index) -> Tuple[Tensor, float]:
        return self.lis_sample[index]

    def __len__(self):
        return len(self.lis_sample)


def custom_collate_fn(batch: List[Tuple[Tensor, float]]):
    inputs = pad_sequence([e[0] for e in batch], batch_first=True)
    outputs = torch.as_tensor([e[1] for e in batch])
    return inputs, outputs


def get_ref_sup_dataloader(ref_sup_sample: List[RefSupSample]):
    # ref_sup_sample = create_lis_sample()
    device: str = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer(model_name_or_path=args.pretrained_sent_bert, device=device)
    lis_data: List[Tuple[Tensor, float]] = []
    with torch.no_grad():
        encoder.eval()
        for sample in ref_sup_sample:
            lis_text: List[str] = [sample.query_text] + [article for article in sample.lis_article]
            lis_encode_vec: Tensor = encoder.encode(lis_text, convert_to_tensor=True, convert_to_numpy=False)
            lis_encode_vec: Tensor = lis_encode_vec.to(torch.device('cpu'))
            lis_data.append((lis_encode_vec, float(sample.label)))
    ref_sup_dataset = RefSupDataset(lis_data)
    del encoder
    # return DataLoader(dataset=ref_sup_dataset, batch_size=None, sampler=None)
    return DataLoader(dataset=ref_sup_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn,
                      num_workers=8)
