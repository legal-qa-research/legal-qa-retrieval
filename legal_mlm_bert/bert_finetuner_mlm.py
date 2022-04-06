import math
from typing import Dict

from datasets import Split, Dataset, DatasetDict, load_dataset
from datasets.arrow_dataset import Batch
from transformers import DataCollatorForLanguageModeling, TrainingArguments, IntervalStrategy, Trainer, AutoTokenizer, \
    AutoModelForMaskedLM

from legal_mlm_bert.args_management import args


class BertFinetunerMLM:
    def __init__(self, args):
        self.args = args
        self.data_path = self.args.corpus_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.args.model_name)
        self.block_size = self.args.max_seq_length
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.num_proc_dataset = 4

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'])

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

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def build_dataset(self):
        raw_dataset: Dataset = load_dataset('text', data_files=self.data_path, split=Split.TRAIN)
        split_raw_dataset: DatasetDict = raw_dataset.train_test_split(test_size=0.1)
        split_raw_dataset['validation'] = split_raw_dataset['test']
        # group_text_dataset = split_raw_dataset.map(
        #     self.group_raw_texts,
        #     batched=True,
        #     num_proc=self.num_proc_dataset
        # )
        tokenized_dataset = split_raw_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.num_proc_dataset,
            remove_columns=split_raw_dataset["train"].column_names,
        )
        group_dataset = tokenized_dataset.map(
            self.group_texts,
            batched=True,
            batch_size=1000,
            num_proc=self.num_proc_dataset,
            remove_columns=tokenized_dataset["train"].column_names,
        )

        return group_dataset

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

        trainer.train(resume_from_checkpoint=self.args.chk_path)
        trainer.save_model('./output_model')

        eval_results = trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


if __name__ == '__main__':
    bert_finetuner = BertFinetunerMLM(args)
    bert_finetuner.start_train()
