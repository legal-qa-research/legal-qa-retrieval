import pickle
from typing import List

import torch
from torch.utils.data import DataLoader

from sbert_hard_neg_filter.constant import hard_neg_path
from utils.constant import pkl_question_pool, pkl_article_pool, pkl_cached_rel
from sent_bert_triploss.data import Data
from sent_bert_triploss.retrieval_evaluator_f2 import RetrievalEvaluatorF2
from sent_bert_triploss.sent_bert_model import get_sent_bert_model
from sentence_transformers import losses
from sentence_transformers import InputExample


class TrainingProcess:
    def __init__(self, args):
        self.args = args
        self.data = Data(pkl_question_pool_path=pkl_question_pool, pkl_article_pool_path=pkl_article_pool,
                         pkl_cached_rel_path=pkl_cached_rel, pkl_cached_split_ids=self.args.split_ids,
                         args=args)
        self.model = get_sent_bert_model(load_chk_point_path=args.load_chk_point)
        if self.args.use_contrast_loss_fn == 1:
            self.loss_fn = losses.OnlineContrastiveLoss(model=self.model)
        else:
            self.loss_fn = losses.CosineSimilarityLoss(model=self.model)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_epoch = args.n_epochs

    def get_evaluator(self, lis_examples: List[InputExample]):
        lis_sentence1 = [examples.texts[0] for examples in lis_examples]
        lis_sentence2 = [examples.texts[1] for examples in lis_examples]
        labels = [examples.label for examples in lis_examples]
        return RetrievalEvaluatorF2(sentences1=lis_sentence1, sentences2=lis_sentence2, labels=labels,
                                    batch_size=self.args.batch_size, show_progress_bar=True,
                                    write_csv=True, name='sent_bert_triploss.csv')

    def start_training(self):
        train_dataloader, lis_test_examples = self.data.build_dataset()
        evaluator = self.get_evaluator(lis_examples=lis_test_examples)
        self.model = self.model.to(self.device)

        self.model.fit(train_objectives=[(train_dataloader, self.loss_fn)], epochs=self.num_epoch,
                       warmup_steps=100, show_progress_bar=True, save_best_model=True,
                       evaluation_steps=self.args.evaluation_steps,
                       checkpoint_path=self.args.chk_point,
                       output_path=self.args.output_path,
                       evaluator=evaluator, scheduler=self.args.scheduler, optimizer_params={'lr': self.args.lr},
                       checkpoint_save_total_limit=self.args.chk_limit
                       )

    def start_train_r2(self):
        train_dataloader, lis_test_examples = self.data.build_dataset()
        evaluator = self.get_evaluator(lis_examples=lis_test_examples)

        r2_train_examples: List[InputExample] = pickle.load(open(hard_neg_path, 'rb'))
        print(f'Number of r2 training examples: {len(r2_train_examples)}')

        r2_train_dataloader = DataLoader(r2_train_examples, shuffle=True, batch_size=self.args.batch_size)

        self.model = self.model.to(self.device)

        self.model.fit(
            train_objectives=[(r2_train_dataloader, self.loss_fn)], epochs=self.num_epoch,
            warmup_steps=100, show_progress_bar=True, save_best_model=True,
            evaluation_steps=self.args.evaluation_steps,
            checkpoint_path=self.args.chk_point,
            output_path=self.args.output_path,
            evaluator=evaluator, scheduler=self.args.scheduler, optimizer_params={'lr': self.args.lr},
            checkpoint_save_total_limit=self.args.chk_limit
        )
