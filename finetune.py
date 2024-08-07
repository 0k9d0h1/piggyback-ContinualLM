#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
from approaches.finetune import Appr
from config import parseing_finetune
from utils import utils
import logging
import os
import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, set_seed, AdamW, T5Tokenizer, LlamaTokenizer
from dataloader.data import get_dataset, get_dataset_eval, get_dataset_t5, get_dataset_t5_eval, dataset_class_num
import random
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)
# Set up logger
logger = logging.getLogger(__name__)

dataset2max_length = {'restaurant_sup': 2,
                      'chemprot_sup': 12,
                      'aclarc_sup': 6,
                      'scierc_sup': 9,
                      'camera_sup': 2,
                      'phone_sup': 2}


def main():
    args = parseing_finetune()
    args = utils.model.prepare_sequence_finetune(args)
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.hyperparameter_tune:
        if 'Llama' in args.base_model_name_or_path:
            wandb.init(project='piggyback_continualDAP_llama_sweep',
                       config=args)
        else:
            wandb.init(project='piggyback_continualDAP_%s_sweep' % (args.base_model_name_or_path),
                       config=args)
    else:
        if 'Llama' in args.base_model_name_or_path:
            wandb.init(project='piggyback_continualDAP_llama',
                       config=args)
        else:
            wandb.init(project='piggyback_continualDAP_%s' % (args.base_model_name_or_path),
                       config=args)
    wandb.run.name = "%s_%s_%s_%d" % (
        args.baseline, args.finetune_type, args.dataset_name, args.seed)
    wandb.run.save()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if args.log_dir is not None:
        handler = logging.FileHandler(args.log_dir)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # if use multilabel-classification datasets:
    if 'multi' in args.dataset_name:
        args.problem_type = 'multi_label_classification'

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets and process the data.
    if "roberta" in args.base_model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(
            args.base_model_name_or_path)
    elif "t5" in args.base_model_name_or_path:
        tokenizer = T5Tokenizer.from_pretrained(args.base_model_name_or_path)
    elif "Llama" in args.base_model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.base_model_name_or_path)
        # Add a new pad token to the tokenizer
        tokenizer.sep_token = '/'
        tokenizer.pad_token = tokenizer.eos_token
    args.tokenizer = tokenizer

    max_length = args.max_seq_length

    logger.info('==> Preparing data..')

    if "roberta" in args.base_model_name_or_path or "Llama" in args.base_model_name_or_path:
        if args.hyperparameter_tune:
            datasets = get_dataset_eval(
                args.dataset_name, tokenizer=tokenizer, args=args)
        else:
            datasets = get_dataset(
                args.dataset_name, tokenizer=tokenizer, args=args)
    elif "t5" in args.base_model_name_or_path:
        if args.hyperparameter_tune:
            datasets = get_dataset_t5_eval(
                args.dataset_name, tokenizer=tokenizer, args=args)
        else:
            datasets = get_dataset_t5(
                args.dataset_name, tokenizer=tokenizer, args=args)
    print(f'Dataset: {args.dataset_name}')

    print(f'Size of training set: {len(datasets["train"])}')
    print(f'Size of testing set: {len(datasets["test"])}')

    train_dataset = datasets['train']
    test_dataset = datasets['test']

    def tokenize_function(examples):
        inputs = examples["text"]
        targets = examples["labels"]
        model_inputs = tokenizer(
            inputs, truncation=True, padding='max_length', max_length=max_length)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, padding='max_length',
                               max_length=dataset2max_length[args.dataset_name])

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if "roberta" in args.base_model_name_or_path or "Llama" in args.base_model_name_or_path:
        test_dataset = test_dataset.map(
            lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
    elif "t5" in args.base_model_name_or_path:
        test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=[
                            'input_ids', 'attention_mask', 'labels'])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=8)

    if "roberta" in args.base_model_name_or_path or "Llama" in args.base_model_name_or_path:
        train_dataset = train_dataset.map(
            lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=max_length), batched=True)
    elif "t5" in args.base_model_name_or_path:
        train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type='torch', columns=[
                             'input_ids', 'attention_mask', 'labels'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=8)  # consider batch size

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}. Decode to: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    if args.class_num is None:
        args.class_num = dataset_class_num[args.dataset_name]

    # Declare the model and set the training parameters.
    logger.info('==> Building model..')

    model = utils.model.lookfor_model_finetune(args)
    # print(model)

    appr = Appr(args)
    appr.train(model, accelerator, train_loader, test_loader)


if __name__ == '__main__':
    main()
