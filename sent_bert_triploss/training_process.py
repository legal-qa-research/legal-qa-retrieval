from typing import List

from sentence_transformers.evaluation import BinaryClassificationEvaluator

from sent_bert_triploss.data import Data
from sent_bert_triploss.sent_bert_model import get_sent_bert_model
from sentence_transformers import losses
from sentence_transformers import InputExample


class TrainingProcess:
    def __init__(self, args):
        self.args = args
        self.data = Data()
        self.model = get_sent_bert_model()
        self.loss_fn = losses.ContrastiveLoss(model=self.model)
        self.num_epoch = args.n_epochs

    def get_evaluator(self, lis_examples: List[InputExample]):
        lis_sentence1 = [examples.texts[0] for examples in lis_examples]
        lis_sentence2 = [examples.texts[1] for examples in lis_examples]
        labels = [examples.label for examples in lis_examples]
        return BinaryClassificationEvaluator(sentences1=lis_sentence1, sentences2=lis_sentence2, labels=labels,
                                             batch_size=self.args.batch_size)

    def start_training(self):
        train_dataloader, lis_test_examples = self.data.build_dataset()
        evaluator = self.get_evaluator(lis_examples=lis_test_examples)

        self.model.fit(train_objectives=[(train_dataloader, self.loss_fn)], epochs=self.num_epoch,
                       warmup_steps=100, show_progress_bar=True, save_best_model=True,
                       checkpoint_path='sent_bert_triploss/chkpoint', output_path='sent_bert_triploss/output_model',
                       evaluator=evaluator)
