from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import pickle
import random
import time
from copy import deepcopy
from test import test

import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
from pyfiglet import Figlet
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from configuration import MODEL_CLASSES, _parse_args
from dataset import NMGDataset
from glue_worker import run_glue
from mask_agent import MaskGenerator
from pretrain_util import evaluate, load_and_cache_examples, train

from nmg_model import GeneratingMasksAC
from transformers import (BertConfig, BertForMaskedLM, BertModel, BertTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                        )

from qa_worker import run_qa
from visualize import visualize

logger = logging.getLogger(__name__)
figlet = Figlet(font='slant')

def _setup_gpu(args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.task_devices = []
    if args.n_gpu >= 2:
        args.device = torch.device("cuda", 0)
        args.device2 = torch.device("cuda", 1)
        for i in range(0, args.n_gpu):
            args.task_devices.append(torch.device("cuda", i))
        args.n_gpu = len(args.task_devices)
    else:
        args.device = device
        args.device2 = device
    return args


def _set_seed(args, latent=0):
    random.seed(args.seed + latent)
    np.random.seed(args.seed + latent)
    torch.manual_seed(args.seed + latent)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed + latent)


def load_neural_mask_generator(args, tokenizer, config_class, base):
    if args.masking == "neural" and not args.continual:
        nmg_path = "bert-base-uncased"
        config = BertConfig.from_pretrained(nmg_path)
        config.model_type = 'bert'
        base_model = BertModel.from_pretrained(nmg_path, config=config)

        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        config.model_version = args.model_version
        nmg_model = GeneratingMasksAC(config=config)
        nmg_model.config = config
        nmg_model.bert = base_model

        if args.device2 is not None: nmg_model.to(args.device2)
        else: nmg_model.to(args.device)
    elif args.masking == "neural" and args.continual:
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.model_type = None
        nmg_model = GeneratingMasksAC(config=config)
        nmg_model.config = config

        if args.device2 is not None: nmg_model.to(args.device2)
        else: nmg_model.to(args.device)
    else:
        config = None
        nmg_model = None

    mask_generator = MaskGenerator(args, nmg_model, config, tokenizer, base=base)

    return mask_generator

def load_language_model(args, config_class, model_cls):
    config = config_class.from_pretrained(args.config_path)
    logger.info(" Load model for Pretraining ")
    model = model_cls.from_pretrained(args.model_weight_path, config=config)
    model = model.to(args.device)
    model.config = config

    return model

def do_pre_training(args, model_cls, train_dataset, mask_generator, config_class,
                    tokenizer, epoch, meta_training, model=None):
    _set_seed(args, latent=epoch)
    if model is None:
        model = load_language_model(args, config_class, model_cls)
    if args.num_train_epochs > 0:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, mask_generator, training=True)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    del model
    del model_to_save
    del train_dataset

def do_fine_tuning(args, run_fn, epoch, task_dataset, meta_training):
    _set_seed(args, latent=epoch)
    logger.info("Run Fine-tuning worker")

    result = run_fn(args, task_dataset,
                    meta_training=meta_training,
                    devices=args.task_devices,)

    task_valid_acc = result

    return task_valid_acc

def do_episode(args, epoch, tokenizer, model_class, config_class,
             run_fn, dataset, mask_generator, meta_training=True,
             model=None):

    _set_seed(args, latent=epoch)
    context_dataset, task_dataset = dataset.sample(args)

    do_pre_training(args, model_class, context_dataset,
            mask_generator, config_class, tokenizer, epoch,
            meta_training, model)

    reward = do_fine_tuning(args, run_fn, epoch,
                            task_dataset=task_dataset,
                            meta_training=meta_training)
    return reward


# Run this function before training
def base_policy_scheduling(mask_generator_base, mask_generator):
    # Switch To Neural Policy Base
    print("Update Neural Policy Baseline")
    mask_generator_base.set_masking_type("neural")
    weights = deepcopy(mask_generator.model.state_dict())
    mask_generator_base.model.load_state_dict(weights)


def main(args):
    if args.mute:
        logging.disable(logging.CRITICAL)

    # Setup logging
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("Process device: %s, n_gpu: %s", args.device, args.n_gpu)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path,
            do_lower_case=args.do_lower_case
    )
    args.pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    print("PAD token idx: {}".format(args.pad_token))

    config = config_class.from_pretrained(args.config_path)
    args.hidden_size = config.hidden_size
    mask_generator = load_neural_mask_generator(args, tokenizer, config_class, base=False)
    mask_generator_base = load_neural_mask_generator(args, tokenizer, config_class,base=True)
    mask_generator.set_masking_type(args.masking)
    # At First, Set Random policy to base mask generator
    mask_generator_base.set_masking_type("random")

    mask_generator_rnd = load_neural_mask_generator(args, tokenizer, config_class, base=True)
    mask_generator_rnd.set_masking_type("random")

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    if args.block_size <= 0: args.block_size = tokenizer.max_len_single_sentence

    if args.task == "qa":
        run_fn = run_qa
    elif args.task == "glue":
        run_fn = run_glue
    else:
        raise NotImplementedError

    base_outer_reward = None
    rnd_outer_reward = None
    _set_seed(args)
    # Task worker does fine-tuning and get reward for mask generator
    mask_generator_base.set_masking_type("random")

    model = load_language_model(args, config_class, model_class) if args.continual else None

    dataset = NMGDataset(args, tokenizer, evaluate=False, meta_training=True)

    context_size = len(dataset.context_dataset)
    init_epoch = max(context_size // args.truncate_pretraining_dataset, 25)
    if args.repeated_init:
        print("Initialize model at every {} epoch".format(init_epoch))

    for epoch in range(args.outer_epoch):
        if args.repeated_init and args.continual and epoch % init_epoch == 0 and epoch > 0:
            model = load_language_model(args, config_class, model_class)

        # If the base is trained, let's change base to neural also
        if args.self_play == "learning" and len(mask_generator.replay_memory) > args.replay_start:
            mask_generator_base.set_masking_type("neural")

        # Inner Loop
        print(figlet.renderText("Episode %d" % epoch))
        print(args.output_dir)

        inputs = {'args': args, 'epoch': epoch, 'tokenizer': tokenizer,
                'model_class': model_class, 'config_class': config_class,
                'run_fn': run_fn,
                'dataset': dataset,
                'model': None,
                }

        ### Random Agent ####
        print("Random Agent")
        inputs['mask_generator'] = mask_generator_rnd
        if args.continual:
            inputs['model'] = deepcopy(model)
        rnd_outer_reward = do_episode(**inputs)

        ### Opponent ###
        if mask_generator_rnd is not None and \
                mask_generator_base.masking_type != "neural":
            base_outer_reward = rnd_outer_reward
            mask_generator_base.action_history = mask_generator_rnd.action_history
        else:
            inputs['mask_generator'] = mask_generator_base
            if args.continual:
                inputs['model'] = deepcopy(model)
            print("Opponent Neural Agent")
            base_outer_reward = do_episode(**inputs)

        ### Player ###
        inputs['mask_generator'] = mask_generator
        if args.continual:
            inputs['model'] = model
        print("Player Neural Agent")
        outer_reward = do_episode(**inputs)


        rnd_history = mask_generator_rnd.action_history

        if args.self_play == "learning":
            mask_generator_base.append_reward_selfplay(op_acc=outer_reward,
                                                       pl_acc=base_outer_reward,
                                                       op_history=mask_generator.action_history,
                                                       rnd_acc=rnd_outer_reward,
                                                       rnd_history=rnd_history,
                                                       )
            mask_generator.append_reward_selfplay(op_acc=base_outer_reward,
                                                  pl_acc=outer_reward,
                                                  op_history=mask_generator_base.action_history,
                                                  rnd_acc=rnd_outer_reward,
                                                  rnd_history=rnd_history,
                                                  )
        elif args.self_play in ["frozen", "random"]:
            mask_generator.append_reward_selfplay(op_acc=base_outer_reward,
                                                  pl_acc=outer_reward,
                                                  op_history=mask_generator_base.action_history,
                                                  rnd_acc=rnd_outer_reward,
                                                  rnd_history=rnd_history)

        else:
            mask_generator.append_reward(base_outer_reward, outer_reward, None)

        if len(mask_generator.replay_memory) > args.replay_start:
            if args.self_play == "learning":
                mask_generator_base.set_masking_type("neural")
            elif args.self_play == "frozen" and outer_reward > base_outer_reward:
                base_policy_scheduling(mask_generator_base,
                                       mask_generator)

        if "neural" in mask_generator.masking_type:
            if args.self_play == "learning":
                mask_generator_base.train_replay(args)
            mask_generator.train_replay(args)
        else:
            mask_generator.write_logs()

        if mask_generator_rnd is not None:
            mask_generator_rnd.initialize_history()
        mask_generator_base.initialize_history()
        mask_generator.initialize_history()
        if "neural" in mask_generator.masking_type:
            mask_generator.save(epoch)
            if args.self_play == "learning":
                mask_generator_base.save(epoch)

    logger.info(" ******* Training Ends ******* ")
    print(mask_generator.acc_history)


if __name__ == "__main__":
    args = _parse_args()
    if args.test:
        test(args)
    else:
        args = _setup_gpu(args)
        _set_seed(args)
        main(args)
