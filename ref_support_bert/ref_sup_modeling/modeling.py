import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.nn import GRU
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.functional import cross_entropy

from ref_support_bert.args_management import RefSupArgument
from transformers.optimization import AdamW
from sklearn.preprocessing import MultiLabelBinarizer


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
        self.label_binarizer = MultiLabelBinarizer()

    def forward(self, model_input: Tensor):
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

    def configure_optimizers(self):
        return AdamW(params=self.parameters(), lr=self.args.lr)
