import os
import clearml

import torch
import transformers
from transformers import HfArgumentParser, Trainer, EarlyStoppingCallback

from helpers import MIN_JP_CODEPOINT, MAX_JP_CODEPOINT, DataArguments, prepare_data, \
    ShibaTrainingArguments, get_model_hyperparams, prepare_clearml_data
from masking import RandomSpanMaskingDataCollator, RandomMaskingDataCollator
from shiba import ShibaForAutoregressiveLanguageModeling, CodepointTokenizer

# cdleong: added these
from clearml import Task 





def main():
    
    
    transformers.logging.set_verbosity_info()
    parser = HfArgumentParser((DataArguments, ShibaTrainingArguments))

    data_args, training_args = parser.parse_args_into_dataclasses()

    # CLEARML TASK SETUP
    Task.add_requirements(
        "tensorboardX", ""
    )
    task = Task.init(
        project_name=training_args.clearml_project_name,
        output_uri=training_args.clearml_output_uri,
        task_name=training_args.clearml_task_name,
    )

    if training_args.clearml_queue_name:
        task.execute_remotely(queue_name=training_args.clearml_queue_name)







    tokenizer = CodepointTokenizer()
    if training_args.masking_type == 'bpe_span':
        print('BPE based-span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, True)
    elif training_args.masking_type == 'rand_span':
        print('Random span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, False)
    elif training_args.masking_type == 'rand_char':
        print('Random character masking')
        # char range: https://stackoverflow.com/a/30200250/4243650
        # we aren't including half width stuff
        
        data_collator = RandomMaskingDataCollator(tokenizer, range(3000, MAX_JP_CODEPOINT))
    else:
        raise RuntimeError('Unknown masking type')


    

    training_args.logging_dir = training_args.output_dir

    # CLEARML data loading 
    if data_args.clearml_training_set and data_args.clearml_validation_set:
        training_data, dev_data = prepare_clearml_data(data_args, training_args)
        print(f"using training_data: {training_data}")
        print(f"using dev_data: {dev_data}")
    else:
        training_data, dev_data = prepare_data(data_args)

    model_hyperparams = get_model_hyperparams(training_args)

    model = ShibaForAutoregressiveLanguageModeling(MAX_JP_CODEPOINT, **model_hyperparams)

    checkpoint_dir = None
    if training_args.resume_from_checkpoint:
        if training_args.load_only_model:
            model.load_state_dict(torch.load(training_args.resume_from_checkpoint))
        else:
            checkpoint_dir = training_args.resume_from_checkpoint
    os.environ['WANDB_PROJECT'] = 'shiba'

    # syntax for EarlyStoppingCallback is shown in 
    # https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
    print(training_args)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=training_data,
                      eval_dataset=dev_data,
                      callbacks=[
                          EarlyStoppingCallback(early_stopping_patience=5),
                      ]
                      )

    trainer.train(resume_from_checkpoint=checkpoint_dir)
    


if __name__ == '__main__':
    main()
