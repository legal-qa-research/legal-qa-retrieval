from datasets import Split, Dataset, DatasetDict, load_dataset
from transformers import PhobertTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, \
    IntervalStrategy, Trainer


class BertFinetunerMLM:
    def __init__(self):
        self.data_path = 'legal_mlm_bert/mini_bert_corpus_path.txt'
        self.tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
        self.model = AutoModelForCausalLM.from_pretrained("vinai/phobert-base")
        self.block_size = 128
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.num_proc_dataset = 1

    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["text"]], padding='max_length', max_length=1024)

    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        result = {
            k: [t[i: i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def build_dataset(self):
        raw_dataset: Dataset = load_dataset('text', data_files=self.data_path, split=Split.TRAIN)
        split_raw_dataset: DatasetDict = raw_dataset.train_test_split(test_size=0.1)
        tokenized_dataset = split_raw_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.num_proc_dataset,
            remove_columns=split_raw_dataset["train"].column_names,
        )
        lm_dataset = tokenized_dataset.map(self.group_texts, batched=True, num_proc=self.num_proc_dataset)
        return lm_dataset

    def start_train(self):
        lm_dataset = self.build_dataset()

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy=IntervalStrategy.EPOCH,
            learning_rate=2e-5,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=lm_dataset["train"],
            eval_dataset=lm_dataset["test"],
            data_collator=self.data_collator,
        )

        trainer.train()


if __name__ == '__main__':
    bert_finetuner = BertFinetunerMLM()
    bert_finetuner.start_train()
