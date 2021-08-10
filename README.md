# Forked Repo!

This repo is a fork of https://github.com/octanove/shiba, edited to allow :
* pretraining and experiment tracking with ClearML, 
* downloading of clearml datasets within a remotely-executed agent, 
* finetuning on https://github.com/cdleong/masakhane-ner 

Files edited: 
* training/train.py
* training/helpers.py
* README.md of course
* requirements.txt: added clearml requirement

Files added: 
* training/finetune_ner.py
* training/finetune_word_segmentation_on_masakhaner.py
* masakhaner_fork_loading_script.py


Running word segmentation on masakhaner using pretrained pytorch_model trained on hf_swahili_no_spaces

```
python finetune_word_segmentation_on_masakhaner.py --output_dir ~/runs/wordseg \
  --resume_from_checkpoint ./hf_swahili_no_spaces_5k_steps/pytorch_model.bin \
  --num_train_epochs 6 \
  --save_strategy no
```


# Licensing notice from original repo

The code and contents of the original repository are provided under the Apache License 2.0. The pretrained model weights are provided under the CC BY-SA 4.0 license. 

# How to cite this work

There is no paper associated with SHIBA, but the repository can be cited like this:

```bibtex
@misc{shiba,
  author = {Joshua Tanner and Masato Hagiwara},
  title = {SHIBA: Japanese CANINE model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/octanove/shiba}},
}
```

Please also cite the original CANINE paper:
```bibtex
@misc{clark2021canine,
      title={CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation}, 
      author={Jonathan H. Clark and Dan Garrette and Iulia Turc and John Wieting},
      year={2021},
      eprint={2103.06874},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
