from typing import List, Tuple, Dict

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from ref_support_bert.ref_sup_data.ref_sup_sample import RefSupSample, SampleGenerator
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


class DataLoaderGenerator:
    def __init__(self):
        device: str = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        self.encoder = SentenceTransformer(model_name_or_path=args.pretrained_sent_bert, device=device)
        self.sample_generator = SampleGenerator()
        self.cache_dict = {}
        self.encoder.eval()

    def get_train_dataloader(self):
        print('Creating train dataloader...')
        t_dl = self.get_ref_sup_dataloader(self.sample_generator.train_examples)
        print('Creating train dataloader Done')
        return t_dl

    def get_lis_test_dataloader(self):
        print('Creating test dataloader ...')
        t_dls = [self.get_ref_sup_dataloader(test_sample) for test_sample in self.sample_generator.test_examples]
        print('Creating train test Dataloader Done')
        return t_dls

    def get_ref_sup_dataloader(self, ref_sup_sample: List[RefSupSample]):
        lis_data: List[Tuple[Tensor, float]] = []
        with torch.no_grad():
            for sample in ref_sup_sample:
                lis_text: List[str] = [sample.query_text] + [article for article in sample.lis_article]
                need_encode_text: List[str] = [t for t in lis_text if t not in self.cache_dict.keys()]
                if len(need_encode_text) > 0:
                    lis_encode_vec: Tensor = self.encoder.encode(need_encode_text, convert_to_tensor=True,
                                                                 convert_to_numpy=False)
                    lis_encode_vec: Tensor = lis_encode_vec.to(torch.device('cpu'))
                    for i, t in enumerate(need_encode_text):
                        self.cache_dict[t] = lis_encode_vec[i]
                seq_encode_vec = torch.stack([self.cache_dict[t] for t in lis_text])
                lis_data.append((seq_encode_vec, float(sample.label)))
        ref_sup_dataset = RefSupDataset(lis_data)
        return DataLoader(dataset=ref_sup_dataset, batch_size=args.batch_size, shuffle=False,
                          collate_fn=custom_collate_fn, num_workers=8)
