import os
from typing import Union, Tuple

import numpy as np
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import nn, Tensor
from torch.nn import GRU
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.functional import binary_cross_entropy
from torch.optim import AdamW

from ref_support_bert.args_management import RefSupArgument

from utils.evaluating_submission import ESP


def execute_f2score(true_pos, total_positive, total_pred_positive):
    precision = true_pos / (total_pred_positive + ESP)
    recall = true_pos / (total_positive + ESP)
    return (5 * precision * recall) / (4 * precision + recall + ESP)


def calculate_f2(predict_label, true_label):
    cnt_true_positive = 0
    total_pred_positive = 0
    total_positive = 0

    for i in range(len(predict_label)):
        if true_label[i] == 1:
            if predict_label[i] == 1:
                cnt_true_positive += 1
            total_positive += 1
        if predict_label[i] == 1:
            total_pred_positive += 1

    return execute_f2score(true_pos=cnt_true_positive, total_positive=total_positive,
                           total_pred_positive=total_pred_positive)


def choose_threshold(test_pred_prob, test_true_label) -> Tuple[float, float]:
    best_threshold = None
    best_f2_score = 0
    for threshold in np.arange(0, 1, 0.01):
        total_f2_score = 0
        for i in range(len(test_pred_prob)):
            pred_prob = test_pred_prob[i]
            true_label = test_true_label[i]
            pred_label = [float(prob >= threshold) for prob in pred_prob]
            total_f2_score += calculate_f2(pred_label, true_label)
        avg_f2_score = total_f2_score / (len(test_pred_prob) + ESP)
        if avg_f2_score > best_f2_score:
            best_threshold = threshold
            best_f2_score = avg_f2_score
    return best_threshold, best_f2_score


def choose_top_k(test_pred_prob, test_true_label) -> Tuple[int, float]:
    best_top_k = None
    best_f2_score = 0
    for top_k in range(1, len(test_pred_prob[0])):

        total_f2_score = 0
        for i in range(len(test_pred_prob)):
            pred_prob = test_pred_prob[i]
            true_label = test_true_label[i]
            sorted_idx = np.flip(np.argsort(pred_prob))
            pred_label = [0 for i in range(len(sorted_idx))]
            for pos_index in sorted_idx[:top_k]:
                pred_label[pos_index] = 1
            total_f2_score += calculate_f2(pred_label, true_label)

        avg_f2_score = total_f2_score / len(test_pred_prob)
        if avg_f2_score > best_f2_score:
            best_f2_score = avg_f2_score
            best_top_k = top_k
    return best_top_k, best_f2_score


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
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.sample_generator = None

    def forward(self, model_input: Tensor) -> Tensor:
        gru_output, gru_hidden = self.recurrent_layer.forward(input=model_input)
        embed_output = gru_output.data[:, -1, :]
        output = self.linear_relu_stack.forward(embed_output)
        return output

    def training_step(self, sample, sample_idx) -> STEP_OUTPUT:
        model_input, label = sample
        model_predict = torch.flatten(self.forward(model_input)).to(torch.device('cpu'))
        loss = binary_cross_entropy(input=model_predict, target=label)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        model_input, label = batch
        model_predict = torch.flatten(self.forward(model_input)).to(torch.device('cpu'))
        return {'dataloader_idx': dataloader_idx, 'predict': model_predict, 'label': label}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:

        test_predict = []
        test_label = []
        for list_batch_output in outputs:
            total_model_predict = []
            total_gold_label = []
            for batch_output in list_batch_output:
                model_predict = batch_output.get('predict')
                label = batch_output.get('label')
                total_model_predict.extend(model_predict)
                total_gold_label.extend(label)
            test_predict.append(total_model_predict)
            test_label.append(total_gold_label)

        best_threshold, f2_score_threshold = choose_threshold(test_predict, test_label)
        best_top_k, f2_score_top_k = choose_top_k(test_predict, test_label)

        output_string = f'best_threshold: {best_threshold} | f2_score_threshold: {f2_score_threshold} |'
        output_string += f'best_top_k: {best_top_k} | f2_score_top_k: {f2_score_top_k} \n'

        f = open(os.path.join(self.args.root_dir, 'validate_result.txt'), 'a')
        f.write(output_string)
        f.close()

        self.log('F2-score', max(f2_score_top_k, f2_score_threshold), on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(params=self.parameters(), lr=self.args.lr)
