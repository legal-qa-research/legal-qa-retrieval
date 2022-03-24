from typing import List

import torch
from sentence_transformers.evaluation import BinaryClassificationEvaluator

from sent_bert_triploss.constant import pkl_question_pool, pkl_article_pool, pkl_cached_rel
from sent_bert_triploss.data import Data
from sent_bert_triploss.sent_bert_model import get_sent_bert_model
from sentence_transformers import losses
from sentence_transformers import InputExample


class TrainingProcess:
    def __init__(self, args):
        self.args = args
        self.data = Data(pkl_question_pool_path=pkl_question_pool, pkl_article_pool_path=pkl_article_pool,
                         pkl_cached_rel_path=pkl_cached_rel)
        self.model = get_sent_bert_model()
        self.loss_fn = losses.ContrastiveLoss(model=self.model)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_epoch = args.n_epochs

    def get_evaluator(self, lis_examples: List[InputExample]):
        lis_sentence1 = [examples.texts[0] for examples in lis_examples]
        lis_sentence2 = [examples.texts[1] for examples in lis_examples]
        labels = [examples.label for examples in lis_examples]
        return BinaryClassificationEvaluator(sentences1=lis_sentence1, sentences2=lis_sentence2, labels=labels,
                                             batch_size=self.args.batch_size, show_progress_bar=True,
                                             write_csv=True, name='sent_bert_triploss.csv')

    def start_training(self):
        train_dataloader, lis_test_examples = self.data.build_dataset()
        evaluator = self.get_evaluator(lis_examples=lis_test_examples)
        self.model = self.model.to(self.device)

        self.model.fit(train_objectives=[(train_dataloader, self.loss_fn)], epochs=self.num_epoch,
                       warmup_steps=100, show_progress_bar=True, save_best_model=True,
                       evaluation_steps=self.args.evaluation_steps,
                       checkpoint_path='sent_bert_triploss/chkpoint',
                       output_path='sent_bert_triploss/output_model',
                       evaluator=evaluator)
