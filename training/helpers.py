import inspect
from dataclasses import dataclass, field
from os import name
from typing import Optional, Dict, Tuple, List

import torch.nn
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, load_dataset
from transformers import TrainingArguments

from shiba import Shiba, CodepointTokenizer


from clearml import Dataset as ClearDataset

# https://stackoverflow.com/questions/19899554/unicode-range-for-japanese/30200250#30200250
MIN_JP_CODEPOINT=3000
MAX_JP_CODEPOINT = 0x9faf
EVAL_DATA_PERCENT = 0.02

#cdleong
from pathlib import Path

@dataclass
class DataArguments:
    data: str = field(
        default=None, metadata={"help": "The location of the Japanese wiki data to use for training."}
    )
    clearml_training_set: Optional[str] = field(
        default=None,
        metadata={
            "help": "The set from which to pull train.txt. Give the name, please, not the ID"
        },
    )

    clearml_validation_set: Optional[str] = field(
        default=None,
        metadata={
            "help": "The set from which to pull validation.txt. Give the name, please, not the ID"
        },
    )

@dataclass
class ShibaTrainingArguments(TrainingArguments):
    masking_type: Optional[str] = field(default='rand_span')
    load_only_model: Optional[bool] = field(default=False)

    group_by_length: Optional[bool] = field(default=True)
    logging_first_step: Optional[bool] = field(default=True)
    learning_rate: Optional[float] = 0.001

    logging_steps: Optional[int] = field(default=200)
    report_to: Optional[List[str]] = field(default_factory=lambda: ['wandb'])
    evaluation_strategy: Optional[str] = field(default='steps')
    fp16: Optional[bool] = field(default=torch.cuda.is_available())
    deepspeed: Optional[bool] = field(default=None)
    warmup_ratio: Optional[float] = 0.025  # from canine

    per_device_eval_batch_size: Optional[int] = field(default=12)
    per_device_train_batch_size: Optional[int] = field(default=12)
    # max that we can fit on one GPU is 12. 12 * 21 * 8 = 2016
    gradient_accumulation_steps: Optional[int] = field(default=21)

    # model arguments - these have to be in training args for the hyperparam search
    dropout: Optional[float] = field(
        default=0.1
    )
    deep_transformer_stack_layers: Optional[int] = field(
        default=12
    )
    local_attention_window: Optional[int] = field(default=128)

    # cdleong: added these
    #############################
    # CLEARML ARGS
    #############################

    clearml_project_name: Optional[str] = field(default="")
    clearml_task_name: Optional[str] = field(default="")
    clearml_output_uri: Optional[str] = field(default="")
    clearml_queue_name: Optional[str] = field(default="")


    clearml_task_name: Optional[str] = field(
        default=None, metadata={"help": "task name for clearML task."}
    )

    clearml_queue: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, will try to execute remotely on the specified queue. aqua-large-gpu0, idx_gandalf_titan-rtx, idx_gandalf_cpu, idx_gandalf_2080ti"
        },
    )


@dataclass
class ShibaWordSegArgs(ShibaTrainingArguments):
    do_predict: Optional[bool] = field(default=True)

    # only used for hyperparameter search
    trials: Optional[int] = field(default=2)
    deepspeed: Optional[bool] = field(default=None)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    report_to: Optional[List[str]] = field(default=lambda: ['tensorboard', 'wandb'])
    num_train_epochs: Optional[int] = 6
    save_strategy: Optional[str] = 'no'

    pretrained_bert: Optional[str] = field(default=None)

        #cdleong
    maskhane_ner_dataset: Optional[str] = field(default="swa")


@dataclass
class ShibaNERArgs(ShibaTrainingArguments):
    '''
    Colin: copied from ShibaWordSegArgs
    '''
    do_predict: Optional[bool] = field(default=True)

    # only used for hyperparameter search
    trials: Optional[int] = field(default=2)
    deepspeed: Optional[bool]  = field(default=None)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    report_to: Optional[List[str]] = field(default=lambda: ['tensorboard'])
    num_train_epochs: Optional[int] = 6
    save_strategy: Optional[str] = 'no'

    pretrained_bert: Optional[str] = field(default=None)

    #cdleong
    maskhane_ner_dataset: Optional[str] = field(default="swa_no_word_boundaries")

@dataclass
class ShibaClassificationArgs(ShibaTrainingArguments):
    do_predict: Optional[bool] = field(default=True)
    eval_steps: Optional[int] = field(default=300)
    logging_steps: Optional[int] = field(default=100)
    learning_rate: Optional[float] = 2e-5
    per_device_train_batch_size: Optional[int] = 6
    num_train_epochs: Optional[int] = 6
    save_strategy: Optional[str] = 'no'

    # only used for hyperparameter search
    trials: Optional[int] = field(default=2)
    deepspeed: Optional[bool] = field(default=None)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    report_to: Optional[List[str]] = field(default=lambda: ['tensorboard', 'wandb'])

    pretrained_bert: Optional[str] = field(default=None)


def get_model_hyperparams(input_args):
    if not isinstance(input_args, dict):
        input_args = input_args.__dict__

    shiba_hyperparams = inspect.getfullargspec(Shiba.__init__).args
    return {key: val for key, val in input_args.items() if key in shiba_hyperparams}


def get_base_shiba_state_dict(state_dict: Dict) -> Dict:
    if sum(1 for x in state_dict.keys() if x.startswith('shiba_model')) > 0:
        return {key[12:]: val for key, val in state_dict.items() if key.startswith('shiba_model')}
    else:
        return state_dict



def prepare_clearml_data(data_args, training_args):

    
    # download the clearML dataset
    training_set_path = ClearDataset.get(dataset_name=data_args.clearml_training_set).get_local_copy()
    validation_set_path = ClearDataset.get(dataset_name=data_args.clearml_validation_set).get_local_copy()

    training_file = training_set_path + "/train.jsonl"    

    training_data = load_dataset('json', data_files=training_file)['train']
    try: 
        validation_file = validation_set_path + "/validation.jsonl"
        dev_data=load_dataset('json', data_files=validation_file)['train']
    except FileNotFoundError:
        dev_file = validation_set_path + "/dev.jsonl"
        dev_data=load_dataset('json', data_files=dev_file)['train']
    return training_data, dev_data

def load_and_fix_state_dict(path_to_pytorch_model:Path):
    '''
    Given the path
    '''
    # we're expecting either a pytorch_model.bin or a whatever.pt
    state_dict = torch.load(path_to_pytorch_model)

    # all the items in the state dict either have "shiba_model." prefixed on them, 
    # or can be safely discarded. See also get_base_shiba_state_dict in helpers.py
    # So, we pull off the prefixes, e.g. shiba_model.whatever just becomes whatever
    # https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494/4 
    # describes a method for pulling the prefixes off
    state_dict_with_fixed_keys = {k.partition("shiba_model.")[2]:state_dict[k] for k in state_dict.keys()}

    # that ends up deleting keys like "autregressive_encoder.norm2.bias" that don't start with "shiba_model."", 
    # reducing them to ""
    # fortunately, we don't _want_ those anyway
    _ = state_dict_with_fixed_keys.pop("", None)

    return state_dict_with_fixed_keys    

def prepare_data(args: DataArguments) -> Tuple[Dataset, Dataset]:
    all_data = load_dataset('json', data_files=args.data)['train']
    data_dict = all_data.train_test_split(train_size=0.98, seed=42)
    training_data = data_dict['train']
    dev_data = data_dict['test']
    return training_data, dev_data    

class SequenceLabelingDataCollator:
    def __init__(self):
        self.tokenizer = CodepointTokenizer()

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad([x['input_ids'] for x in batch])
        input_ids = padded_batch['input_ids']
        attention_mask = padded_batch['attention_mask']

        # don't compute loss from padding
        labels = pad_sequence([torch.tensor(x['labels']) for x in batch], batch_first=True, padding_value=-100)
        # also don't compute loss from CLS or SEP tokens
        special_token_mask = (input_ids == self.tokenizer.CLS) | (input_ids == self.tokenizer.SEP)
        labels = labels.where(~special_token_mask, torch.full(labels.shape, -100))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ClassificationDataCollator:
    def __init__(self):
        self.tokenizer = CodepointTokenizer()

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad([x['input_ids'] for x in batch])
        input_ids = padded_batch['input_ids']
        attention_mask = padded_batch['attention_mask']

        labels = torch.tensor([x['labels'] for x in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
