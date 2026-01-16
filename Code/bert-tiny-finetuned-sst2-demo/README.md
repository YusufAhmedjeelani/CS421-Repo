---
library_name: transformers
license: mit
base_model: prajjwal1/bert-tiny
tags:
- generated_from_trainer
model-index:
- name: bert-tiny-finetuned-sst2-demo
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-tiny-finetuned-sst2-demo

This model is a fine-tuned version of [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 63   | 0.6841          |


### Framework versions

- Transformers 4.57.0
- Pytorch 2.8.0
- Datasets 4.1.1
- Tokenizers 0.22.1
