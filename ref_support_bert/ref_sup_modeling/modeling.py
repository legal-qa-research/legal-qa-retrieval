from typing import Dict

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS, EPOCH_OUTPUT
from torch import nn, Tensor
from torch.nn import GRU
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.functional import cross_entropy
from torch.optim import AdamW

from ref_support_bert.args_management import RefSupArgument

from ref_support_bert.ref_sup_data.ref_sup_dataloader import get_ref_sup_dataloader
from ref_support_bert.ref_sup_data.ref_sup_sample import SampleGenerator
from utils.evaluating_submission import ESP


class RefSupModel(LightningModule):
    def __init__(self, input_size, args: RefSupArgument):
        super(RefSupModel, self).__init__()
        self.args = args
        self.recurrent_layer = GRU(input_size=input_size, hidden_size=input_size, num_layers=2)
        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )
        self.sample_generator = None

    def prepare_data(self) -> None:
        self.sample_generator = SampleGenerator()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.sample_generator is not None, 'Data is not prepared'
        return get_ref_sup_dataloader(self.sample_generator.train_examples)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.sample_generator is not None, 'Data is not prepared'
        return [get_ref_sup_dataloader(test_batch) for test_batch in self.sample_generator.test_examples]

    def forward(self, model_input: Tensor) -> Tensor:
        gru_output, gru_hidden = self.recurrent_layer.forward(input=model_input)
        embed_output = gru_output.data[:, -1, :]
        output = self.linear_relu_stack.forward(embed_output)
        return output

    def training_step(self, sample, sample_idx) -> STEP_OUTPUT:
        model_input, label = sample
        model_predict = self.forward(model_input)
        label_vec = torch.stack(
            [torch.as_tensor([float(spec_label == 0), float(spec_label == 1)]) for spec_label in label])
        loss = cross_entropy(input=model_predict, target=label_vec)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        model_input, label = batch
        model_predict = self.forward(model_input)
        return {'dataloader_idx': dataloader_idx, 'predict': model_predict, 'label': label}

    def count_predict(self, predict, label):
        cnt_true_positive = 0
        total_pred_positive = 0
        total_positive = 0

        for i in range(len(predict)):
            pred_label = float(predict[i][0] < predict[i][1])
            true_label = label[i]
            if true_label == 1:
                if pred_label == 1:
                    cnt_true_positive += 1
                total_positive += 1
            if pred_label == 1:
                total_pred_positive += 1

        return cnt_true_positive, total_positive, total_pred_positive

    def cal_f2score(self, true_pos, total_positive, total_pred_positive):
        precision = true_pos / (total_pred_positive + ESP)
        recall = true_pos / (total_positive + ESP)
        return (5 * precision * recall) / (4 * precision + recall + ESP)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        f2score = {}
        total_f2score = 0

        for list_batch_output in outputs:
            cnt_true_positive = 0
            cnt_total_positive = 0
            cnt_total_pred_positive = 0
            for batch_output in list_batch_output:
                model_predict = batch_output.get('predict')
                label = batch_output.get('label')
                true_positive, total_positive, total_pred_positive = self.count_predict(predict=model_predict,
                                                                                        label=label)
                cnt_true_positive += true_positive
                cnt_total_positive += total_positive
                cnt_total_pred_positive += total_pred_positive
            total_f2score += self.cal_f2score(cnt_true_positive, cnt_total_positive, cnt_total_pred_positive)
        avg_f2score = total_f2score / len(outputs)
        self.log('F2-score', avg_f2score, on_epoch=True, prog_bar=True, logger=True)

        # for output in outputs:
        #     dataloader_idx = output.get('dataloader_idx')
        #     if dataloader_idx is None:
        #         dataloader_idx = 0
        #     model_predict = output.get('predict')
        #     label = output.get('label')
        #     true_positive, total_positive, total_pred_positive = self.count_predict(predict=model_predict, label=label)
        #     if dataloader_idx in f2score.keys():
        #         f2score[dataloader_idx]['true_pos'] += true_positive
        #         f2score[dataloader_idx]['total_positive'] += total_positive
        #         f2score[dataloader_idx]['total_pred_positive'] += total_pred_positive
        #     else:
        #         f2score[dataloader_idx]['true_positive'] = true_positive
        #         f2score[dataloader_idx]['total_positive'] = total_positive
        #         f2score[dataloader_idx]['total_pred_positive'] = total_pred_positive
        #
        # total_f2score = 0
        # cnt_ques = len(f2score.keys())
        # for dataloader_idx in f2score.keys():
        #     total_f2score += self.cal_f2score(f2score[dataloader_idx])

    def configure_optimizers(self):
        return AdamW(params=self.parameters(), lr=self.args.lr)
