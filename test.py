# This code is for testing the performance of trained neural mask generator
# Overall desing is similar to main.py, but I seperate this for better reproducing.

import argparse
import torch
import os
import logging
from pprint import pprint

from transformers import (BertConfig, BertForMaskedLM, BertTokenizer, BertModel,
                          DistilBertConfig,
                          DistilBertForMaskedLM,
                          DistilBertTokenizer)
from nmg_model import GeneratingMasksAC

from pretrain_util import load_and_cache_examples, train
from mask_agent import MaskGenerator

import torch
import torch.multiprocessing as mp
import numpy as np
import random
from dataset import NMGDataset

from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
        "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
        }

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def test(args):
    assert args.checkpoint is not None

    loaded_args = torch.load(os.path.join(args.checkpoint, 'training_args.bin')).__dict__
    # To avoid accident
    delete_keys = ['output_dir', 'max_steps', 'per_gpu_train_batch_size',
                   'per_gpu_eval_batch_size', 'model_type',
                   'masking', 'mute', 'num_train_epochs',
                   'num_train_task_epochs', 'masking_prob',
                   'truncate_task_dataset',
                   'truncate_pretraining_dataset', 'seed',
                   'local_rank', 'checkpoint', 'log_file',
                   'mask_dir', 'stochastic']
    for key in delete_keys:
        if key in loaded_args.keys():
            del loaded_args[key]
    args.__dict__.update(loaded_args)

    if args.local_rank in [-1, 0]:
        pprint(args.__dict__)

    if args.mute:
        logging.disable(logging.CRITICAL)

    # Setup CUDA, GPU distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else: # initialize the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    args.device2 = None

    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("Process device: %s, n_pu: %s", device, args.n_gpu)

    tb_writer = SummaryWriter("logs")

    logger.info("Testing parameters %s", args)

    set_seed(args)

    assert args.config_path is not None
    assert args.tokenizer_path is not None
    assert args.model_weight_path is not None

    print(args.config_path)
    print(args.tokenizer_path)
    print(args.model_weight_path)
    print(args.do_lower_case)

    logging.warning("Process %d: Loading base config, tokenizer, mask_model", args.local_rank)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path,
            do_lower_case=args.do_lower_case
    )
    args.pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    print("PAD token idx: {}".format(args.pad_token))
    config = config_class.from_pretrained(args.config_path)

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence

    if args.masking == "neural" and not args.continual:
        mask_config = BertConfig.from_pretrained(os.path.join(args.checkpoint, args.mask_dir))
        mask_config.model_type = 'bert'

        mask_config.attention_probs_dropout_prob = 0.0
        mask_config.hidden_dropout_prob = 0.0
        mask_config.model_version = args.model_version
        mask_model = GeneratingMasksAC.from_pretrained(os.path.join(args.checkpoint, args.mask_dir), config=mask_config)

        mask_model = mask_model.to(args.device)
        mask_model.eval()

    elif args.masking == "neural" and args.continual:
        mask_config = BertConfig.from_pretrained(os.path.join(args.checkpoint, args.mask_dir))
        mask_config.model_type = None
        mask_config.attention_probs_dropout_prob = 0.0
        mask_config.hidden_dropout_prob = 0.0

        archive_path = os.path.join(args.checkpoint, args.mask_dir, "pytorch_model.bin")

        mask_model = GeneratingMasksAC(config=mask_config)
        if args.local_rank != -1:
            map_location = 'cpu'
        else:
            map_location = None
        mask_model.load_state_dict(
                torch.load(archive_path, map_location=map_location))

        mask_model = mask_model.to(args.device)
        mask_model.eval()
    else:
        mask_model = None

    logger.warning("Process %d: Building Mask Generator", args.local_rank)

    mask_generator = MaskGenerator(args, mask_model, config, tokenizer, False)

    if args.masking == "nopt":
        args.num_train_epochs = 0
        args.masking = "random"
    mask_generator.set_masking_type(args.masking)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    logger.warning("Process %d: Load Datasets", args.local_rank)
    dataset = NMGDataset(args, tokenizer, evaluate=False, meta_training=False)

    # Load Dataset
    context_dataset, task_dataset = dataset.get(args)

    # Block1
    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.warning("Process %d: Building Task Worker", args.local_rank)
    if args.task == "qa":
        from qa_worker import run_qa
        run_fn = run_qa
    elif args.task == "glue":
        from glue_worker import run_glue
        run_fn = run_glue
    else:
        raise NotImplementedError

    logger.warning("Process %d: Load Language Model", args.local_rank)
    config = config_class.from_pretrained(args.config_path)

    model = model_class.from_pretrained(args.model_weight_path, config=config)
    model = model.to(args.device)
    model.config = config

    logger.warning("Process %d: Run Pre-Training", args.local_rank)
    if args.num_train_epochs > 0:
        global_step, tr_loss = train(args, context_dataset, model, tokenizer,
                mask_generator, meta_training=False)
    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    logger.warning("Process %d: Run fine-tuning Worker", args.local_rank)
    # Run Fine-Tuning Worker
    logger.info("Run Fine-Tuning Worker")

    result = run_fn(args, task_dataset, meta_training=False)

    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.task == "glue":
            task_valid_acc = result["acc_"]
            if "f1_"in result.keys():
                task_valid_f1 = result["f1_"]
            else:
                task_valid_f1 = task_valid_acc
        elif args.task == "qa":
            task_valid_acc = result["f1_"]
            task_valid_f1 = result["f1_"]
            task_valid_exact = result["exact_"]

        logger.info(" **** Final Fine-Tuning Results : %.4f **** " % task_valid_acc)

    logger.info("Testing parameters %s", args)

    return