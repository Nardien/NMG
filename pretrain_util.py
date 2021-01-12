from __future__ import absolute_import, division, print_function

import tqdm
import json
import torch
from torch import nn
import numpy as np
import logging
import argparse
import glob
import os
import pickle
import random
from pyfiglet import Figlet
import time
import shutil
import math

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForMaskedLM, BertTokenizer,)

from mask_agent import MaskGenerator
logger = logging.getLogger(__name__)
figlet = Figlet(font='slant')


class TextDatasetSingle(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512, truncate=-1, model_type='bert', episode=-1):
        assert os.path.isfile(file_path)
        block_size = min(510, block_size)
        directory, file_name = os.path.split(file_path)
        cached_features_file = os.path.join(directory,
                'cached_lm_{}_{}_{}'.format(block_size, file_name, model_type))

        # add_special_tokens = tokenizer.add_special_tokens_single_sentence
        add_special_tokens = tokenizer.add_special_tokens

        print(cached_features_file)
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text_corpus = f.readlines()

            tokenized_text_corpus = dict()
            for idx, text in enumerate(text_corpus):
                tokenized_text_corpus[idx] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            self.examples = {str(i):[] for i in range(len(tokenized_text_corpus))}
            total_tokenized_size = 0
            total_len = 0
            i = 0
            pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
            for text_idx, tokenized_text in tokenized_text_corpus.items():
                total_tokenized_size += len(tokenized_text)
                total_len += 1

                # Flags for avoiding using remaining tail of context as new context
                tail = False
                while len(tokenized_text) >= block_size:
                    tail = True
                    print(str(i) + " ", end='')
                    input_text = add_special_tokens(tokenized_text[:block_size])
                    self.examples[str(text_idx)].append(input_text)
                    tokenized_text = tokenized_text[block_size:]
                    i += 1

                if len(tokenized_text) < block_size and not tail:
                    print(str(i) + " ", end='')
                    if model_type in ['albert', 't5']:
                        input_text = tokenized_text
                        while len(input_text) < block_size:
                            input_text.append(pad_token)
                    else:
                        input_text = add_special_tokens(tokenized_text)
                        while len(input_text) < block_size + 2:
                            input_text.append(pad_token)
                    self.examples[str(text_idx)].append(input_text)
                    i += 1

            print("\n{}".format(total_tokenized_size / total_len))
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Total number of contexts --> {}'.format(len(self.examples)))

        sampled_keys, examples = [], []
        for example in self.examples.items():
            idx, context = example
            examples += context
            sampled_keys.append(str(idx))

        self.sampled_keys = sampled_keys
        self.sampled_examples = examples

        # For meta-test
        self.train_sampled_keys = sampled_keys
        self.train_sampled_examples = examples

    # Arbitrary Split given context set to train / valid to avoid overlap
    def split(self, train_keys=None, valid_keys=None):
        # Only for Meta-Training
        if train_keys is None and valid_keys is None:
            keys = list(self.examples.keys())
            random.shuffle(keys)
            train_num = int(len(keys) * 0.9)
            train_keys, valid_keys = keys[:train_num], keys[train_num:]


        self.valid_examples = {key : self.examples[key] for key in valid_keys}
        self.train_examples = {key : self.examples[key] for key in train_keys}

        self.valid_sampled_keys = [str(idx) for idx in valid_keys]
        self.train_sampled_keys = [str(idx) for idx in train_keys]

    def sample(self, truncate, episode=-1):
        if truncate > 0 and truncate < len(self.train_examples):
            examples = random.sample(self.train_examples.items(), truncate)
        else:
            examples = self.train_examples.items()
        sampled_keys, sampled_examples = [], []
        for example in examples:
            idx, context = example
            sampled_examples += context
            sampled_keys.append(str(idx))
        # To Avoid Too Many Sub-Context
        if truncate > 0 and truncate < len(sampled_examples):
            sampled_examples = random.sample(sampled_examples, truncate)
        self.train_sampled_examples = sampled_examples
        self.train_sampled_keys = sampled_keys

    def __len__(self):
        return len(self.train_sampled_examples)

    def __getitem__(self, item):
        return torch.tensor(self.train_sampled_examples[item], dtype=torch.long)


class TextDatasetSingleStatic(TextDatasetSingle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = []
        self.inputs = []

    def add_label(self, mask_generator, tokenizer, args, model=None):
        assert len(self.labels) == 0 and len(self.inputs) == 0
        for example in self.train_sampled_examples:
            example = torch.tensor(example, dtype=torch.long)
            inputs, labels = mask_generator.mask(example, tokenizer, args, model=model)
            if len(inputs.shape) > 1:
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)
            self.inputs.append(inputs)
            self.labels.append(labels)

    def sample(self, truncate, episode=-1):
        self.labels = []
        self.inputs = []
        super().sample(truncate, episode)

    def __len__(self):
        return len(self.train_sampled_examples)

    def __getitem__(self, item):
        return (self.inputs[item], self.labels[item])


def load_and_cache_examples(args, tokenizer, evaluate=False, meta_training=True, episode=-1):
    start = time.time()
    if meta_training:
        dataset = TextDatasetSingleStatic(tokenizer, file_path=args.eval_data_file if
                evaluate else args.train_data_file, block_size=args.block_size,
                truncate=args.truncate_pretraining_dataset,
                model_type=args.model_type, episode=episode)
    else:
        dataset = TextDatasetSingle(tokenizer, file_path=args.eval_data_file if
                evaluate else args.train_data_file, block_size=args.block_size,
                truncate=args.truncate_pretraining_dataset,
                model_type=args.model_type, episode=episode)
    print("Elapsed Time for loading context: {} sec".format(time.time()-start))
    return dataset

def train(args, train_dataset, model, tokenizer, mask_generator, training=False, meta_training=True):
    """ Train the model"""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if meta_training:
        args.save_history = True
        train_dataset.add_label(mask_generator, tokenizer, args, model)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, args.task_devices, output_device=0)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_acc, logging_acc = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Pre-Training", disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if epoch > 0: args.save_history = False
            else: args.save_history = True
            if meta_training:
                inputs, labels = batch
            else:
                inputs, labels = mask_generator.mask(batch, tokenizer, args,
                        model=model.module if hasattr(model, 'module') else model)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            input_mask = ~inputs.eq(args.pad_token)
            model.train()
            _inputs = {'input_ids':        inputs,
                       'masked_lm_labels': labels,
                       'attention_mask': input_mask,}

            outputs = model(**_inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        # print(loss)
        # print("Global step: {} / Training Loss: {}".format(global_step, tr_loss / global_step))
        # print("Global Accuracy: {} / Training Accuracy: {}".format(global_step, tr_acc / global_step))
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    del optimizer
    del scheduler
    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, mask_generator, prefix=""):
    eval_output_dir = args.output_dir

    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if not args.mute:
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(args.device)
        inputs, labels = mask_generator.mask(batch, tokenizer, args)
        # inputs, labels = mask_tokens(batch, tokenizer, args)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        input_mask = ~inputs.eq(args.pad_token)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels,
                            attention_mask=input_mask)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results

