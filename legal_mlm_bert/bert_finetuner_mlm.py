import math
from typing import Dict

from datasets import Split, Dataset, DatasetDict, load_dataset
from datasets.arrow_dataset import Batch
from transformers import DataCollatorForLanguageModeling, TrainingArguments, IntervalStrategy, Trainer, AutoTokenizer, \
    RobertaForMaskedLM

from legal_mlm_bert.args_management import args


class BertFinetunerMLM:
    def __init__(self, args):
        self.args = args
        self.data_path = self.args.corpus_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name, use_fast=True)
        self.model = RobertaForMaskedLM.from_pretrained(self.args.model_name)
        self.block_size = self.args.max_seq_length
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.num_proc_dataset = 4

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding='max_length', max_length=self.block_size)

    def group_raw_texts(self, examples: Batch) -> Dict:
        assert 'text' in list(examples.keys()), 'Expected text key in group_raw_text method'
        concatenated_txt = sum([txt.split(' ') for txt in examples['text']], [])
        total_length = len(concatenated_txt)
        result = {
            'text': [
                ' '.join(concatenated_txt[i: i + self.block_size]) for i in range(0, total_length, self.block_size)
            ]
        }
        return result

    def build_dataset(self):
        raw_dataset: Dataset = load_dataset('text', data_files=self.data_path, split=Split.TRAIN)
        split_raw_dataset: DatasetDict = raw_dataset.train_test_split(test_size=0.1)
        group_text_dataset = split_raw_dataset.map(
            self.group_raw_texts,
            batched=True,
            num_proc=self.num_proc_dataset
        )
        lm_dataset = group_text_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.num_proc_dataset,
            remove_columns=split_raw_dataset["train"].column_names,
        )
        return lm_dataset

    def start_train(self):
        lm_dataset = self.build_dataset()

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy=IntervalStrategy.EPOCH,
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=1
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=lm_dataset["train"],
            eval_dataset=lm_dataset["test"],
            data_collator=self.data_collator,
        )

        trainer.train()

        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        trainer.save_model('./output_model')


if __name__ == '__main__':
    bert_finetuner = BertFinetunerMLM(args)
    bert_finetuner.start_train()
