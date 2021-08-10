import os
from typing import Dict

import torch
import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, EvalPrediction, BertForTokenClassification, AutoTokenizer, \
    DataCollatorForTokenClassification

from shiba import CodepointTokenizer, ShibaForSequenceLabeling
from helpers import get_model_hyperparams, SequenceLabelingDataCollator, \
    ShibaNERArgs, get_base_shiba_state_dict, load_and_fix_state_dict
import torchmetrics

#cdleong added these imports
from pathlib import Path
from clearml import Task, Dataset

#TODO: adapt this to NER! 
# * [x] vocab size/class count
# * [] process_examples needs fixing
# * [] clearml integration: init task
# * [] clearml integration: download dataset from masakhaNER fork
# * [] clearml integration: download pretrained model from clearML






def main():
    transformers.logging.set_verbosity_info()
    parser = HfArgumentParser((ShibaNERArgs,))

    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.logging_dir = training_args.output_dir

    if training_args.pretrained_bert is None:

        tokenizer = CodepointTokenizer()

        # def process_example(example: Dict) -> Dict:
        #     tokens = example['tokens']  # cdleong: should be a list of characters
        #     text = ''.join(tokens)  # cdleong: combines the list of characters to one big string. 
        #     labels = [0]  # CLS token
        #     for token in tokens:
        #         new_label_count = len(token)
        #         new_labels = [1] + [0] * (new_label_count - 1)
        #         labels.extend(new_labels)

        #     input_ids = tokenizer.encode(text)['input_ids']

        #     return {
        #         'input_ids': input_ids,
        #         'labels': labels
        #     }

        def process_example(example: Dict) -> Dict:
            '''Designed to process MasakhaNER examples from our fork at https://github.com/cdleong/masakhane-ner
            Based on process_example commented out above, but specifically pulling the "ner_tags" field
            See also https://github.com/cdleong/masakhane-ner/blob/main/custom_huggingface_loading_script.py
            '''
            original_data = example['tokens']
            text = ''.join(original_data)  # join all the character together 

            # 
            tokenized_text = tokenizer.encode(text) # returns a dict with keys "input_ids" and "attention_mask"... I think
            input_ids = tokenized_text['input_ids']  # pull out the input_ids, which should be ints

            labels = [0] # make a list with "O" (outside) at the start, because tokenizer adds a CLS token at the beginning

            masakhaNERlabels=example["ner_tags"]  # masakhaNER loading script gives us ints, we just rename them
            labels.extend(masakhaNERlabels) # now the input_ids and labels are the same length!


            return {
            'input_ids': input_ids,
            'labels': labels
            }

        model_hyperparams = get_model_hyperparams(training_args)

        
        # label_count=2 # beginning of word or not beginning of word are the two labels for word segmentation
        
        # MasakhaNER has 9 possible labels. 
        # There are 4 entity types: [ORG, PER, DATE, LOC] entities
        # Each token is annotated as either B-{entity label} to mark the beginning of an entity, 
        # or I-{entity label} to mark the continuation, 
        # or O for outside/not part of an entity. So possible labels include B-ORG, I-ORG, B-PER, O...
        # Example from MasakhaNER swa dev set: 
        # Masoko O
        # ya O
        # Japan B-LOC
        # , O
        # China B-LOC
        # na O
        # Thailand B-LOC
        # yalikuwa O 
        # 
        # Another example: 
        # Mike B-PER
        # Pompeo I-PER       
        masakhane_ner_label_count=9 
        label_count=masakhane_ner_label_count
        model = ShibaForSequenceLabeling(label_count, **model_hyperparams)

        if training_args.resume_from_checkpoint:
            print('Loading and using base shiba states from', training_args.resume_from_checkpoint)
            checkpoint_state_dict = torch.load(training_args.resume_from_checkpoint)
            model.shiba_model.load_state_dict(get_base_shiba_state_dict(checkpoint_state_dict))

        data_collator = SequenceLabelingDataCollator()
    else:
        model = BertForTokenClassification.from_pretrained(training_args.pretrained_bert, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(training_args.pretrained_bert)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

        def process_example(example: Dict) -> Dict:
            tokens = example['tokens']  # cdleong: should be a list of characters
            text = ''.join(tokens)  # cdleong: combines the list of characters to one big string. 
            labels = [0]  # CLS token
            for token in tokens:
                new_label_count = len(token)
                new_labels = [1] + [0] * (new_label_count - 1)
                labels.extend(new_labels)

            encoded_example = tokenizer(text, truncation=True)
            encoded_example['labels'] = labels

            return encoded_example

    #cdleong: old code
    # dep  = load_dataset('universal_dependencies', 'ja_gsd')
    # dep = dep.map(process_example, remove_columns=list(ner_dataset['train'][0].keys()))

    # cdleong: new code, using the masakhaNER loading script we've created. 
    masakhaner_dataset = load_dataset("./masakhaner_fork_loading_script.py", 
    training_args.maskhane_ner_dataset,
    ) 
    ner_dataset = masakhaner_dataset.map(process_example, remove_columns=list(masakhaner_dataset['train'][0].keys()))

    # os.environ['WANDB_PROJECT'] = 'shiba'

    def compute_metrics(pred: EvalPrediction) -> Dict:

        if training_args.pretrained_bert is None:
            label_probs, embeddings = pred.predictions
            labels = torch.tensor(pred.label_ids)
            predictions = torch.max(torch.exp(torch.tensor(label_probs)), dim=2)[1]
        else:
            predictions = torch.max(torch.tensor(pred.predictions), dim=2)[1]
            labels = torch.tensor(pred.label_ids)

        metric = torchmetrics.F1(multiclass=False)
        for label_row, prediction_row in zip(labels, predictions):
            row_labels = []
            row_predictions = []
            for lbl, pred in zip(label_row, prediction_row):
                if lbl != -100:
                    row_labels.append(lbl)
                    row_predictions.append(pred)

            row_labels = torch.tensor(row_labels)
            row_predictions = torch.tensor(row_predictions)
            assert row_labels.shape == row_predictions.shape
            metric.update(row_predictions, row_labels)

        f1 = metric.compute()

        return {
            'f1': f1.item()
        }

    print(training_args)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=ner_dataset['train'],
                      eval_dataset=ner_dataset['validation'],
                      compute_metrics=compute_metrics
                      )

    trainer.train()
    posttrain_metrics = trainer.predict(ner_dataset['test']).metrics
    print(posttrain_metrics)


if __name__ == '__main__':
    main()
