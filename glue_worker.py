
import os
import copy
import argparse
import logging
import torch
import numpy as np
import random
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                             TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from glue_util import glue_processors as processors
from glue_util import glue_output_modes as output_modes
from glue_util import convert_examples_to_features, compute_metrics

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForSequenceClassification, BertTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer,)
import pickle
import time

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        }

def argument_reset(args_origin):
    args = copy.deepcopy(args_origin)

    args.task_name = args.glue_dataset
    args.output_dir = os.path.join(args_origin.output_dir, args.task_name.upper())
    args.data_dir = "/st2/mkkang/dataset/glue_data/" + args.task_name.upper()
    args.model_name_or_path = args_origin.output_dir
    args.max_seq_length = 128
    if args.task_name == "imdb":
        args.max_seq_length = 256

    args.logging_steps = args_origin.task_logging_steps
    args.save_steps = args_origin.task_save_steps
    args.truncate_task_dataset = args_origin.truncate_task_dataset
    args.overwrite_output_dir = True
    args.num_train_epochs = args_origin.num_train_task_epochs
    args.max_steps = args.task_max_steps
    args.learning_rate = 2e-5
    return args


def train(args, train_dataset, model, tokenizer, meta_training):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if meta_training:
        test_dataset = train_dataset[1]
        train_dataset = train_dataset[0]

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
        model = torch.nn.DataParallel(model, args.devices)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    if not args.mute:
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
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}

            if args.model_type == 'distilbert':
                del inputs['token_type_ids']

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    te_loss = 0.0
    acc = 0.0
    if meta_training:
        test_step = 0
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)

        # Test!
        all_results = []
        num_samples = 0
        preds = None
        for batch in tqdm(test_dataloader, desc="Testing"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3]}
                if args.model_type == 'distilbert':
                    del inputs['token_type_ids']

                outputs = model(**inputs)
                loss, logits = outputs[:2]  # model outputs are always tuple in pytorch-transformers (see doc)
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training

                te_loss += loss.item()
            test_step += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        te_loss = te_loss / test_step
        preds = np.argmax(preds, axis=1)
        eval_task = args.task_name
        result = compute_metrics(eval_task, preds, out_label_ids)
        acc = result['acc'] * 100
    return global_step, tr_loss / global_step, acc


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = args.eval_batch_size)

        # Eval!
        logger.info(" ***** Running evaluation {} ***** ".format(prefix))
        logger.info("  Num examples = %d  ", len(eval_dataset))
        logger.info("  Batch size = %d  ", args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels':         batch[3]}
                if args.model_type == 'distilbert':
                    del inputs['token_type_ids']

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        print(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info(" %s = %s", key, str(result[key]))
                writer.write(" %s = %s\n" % (key, str(result[key])))

    return results

def load_examples(args, tokenizer, evaluate=False, meta_training=False, sampled_keys=None):
    args = argument_reset(args)

    task = args.task_name

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    model_name = args.model_type

    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        model_name,
        str(args.max_seq_length),
        str(task)))

    print("Load {}".format(cached_features_file))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=False if args.task == "chemical" else tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )

        if not evaluate:
            max_id = features[-1].ex_index
            tmp_features = {str(i):[] for i in range(max_id+1)}
            for feature in features:
                context_id = feature.ex_index
                tmp_features[str(context_id)].append(feature)
            features = tmp_features

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as f:
                pickle.dump(features, f)

    return cached_features_file, features, None

def load_and_cache_examples(args, task, tokenizer, evaluate=False, meta_training=False, sampled_keys=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    model_name = args.model_type

    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        model_name,
        str(args.max_seq_length),
        str(task)))

    print("Load {}".format(cached_features_file))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=False if args.task == "chemical" else tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as f:
                pickle.dump(features, f)

    if not evaluate and args.matching_sample:
        train_features = []
        sampled_keys = list(set(sampled_keys))
        for key in sampled_keys:
            feature = features[int(key)]
            train_features.append(feature)

        train_feature_length = len(train_features) if args.truncate_task_dataset < 0 \
                               else min(args.truncate_task_dataset, len(train_features))
        extracted_features = random.sample(features, len(features))
        test_features = []
        count = 0
        for feature in extracted_features:
            if count >= train_feature_length:
                break
            test_features.append(feature)
            count += 1
        test_features = random.sample(test_features, len(test_features))
        test_features = test_features[:train_feature_length]
        if args.truncate_task_dataset > 0:
            train_features = random.sample(train_features, train_feature_length)
        print('Extracted train features --> %s' % len(train_features))
        print('Extracted test features --> %s' % len(test_features))

        if not meta_training:
            features = train_features
    elif not evaluate:
        all_features = []
        for feature in features:
            all_features.append(feature)
        features = all_features

        if args.truncate_task_dataset > 0:
            features = features[:args.truncate_task_dataset]

        train_feature_length = int(0.8 * len(features))
        train_features = features[:train_feature_length]
        test_features = features[train_feature_length:]

        print('Extracted train features --> %s' % len(train_features))
        print('Extracted test features --> %s' % len(test_features))

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if not meta_training:
        if args.truncate_task_dataset > 0 and not evaluate:
            features = features[:args.truncate_task_dataset]
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    else:
       dataset = []
       for features in [train_features, test_features]:
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            if output_mode == "classification":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            elif output_mode == "regression":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

            dataset_ = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            dataset.append(dataset_)

    return dataset

# CUDA_MAGIC -> avoid cuda gpu exploration
def cuda_magic():
    start = time.time()
    tmp = "./dataset/mrqa/cached_newsqa_train_bert-base-uncased_384_MS"
    with open(tmp, 'rb') as f:
        _tmp = pickle.load(f)
    del _tmp
    print("Elapsed Time for CUDA Magic... {} sec".format(time.time() - start))

def run_glue(args, train_dataset, meta_training, reset_env=True, devices=[], sampled_keys=None):
    # Reallocate argument for Task
    args = argument_reset(args)

    if not args.overwrite_output_dir and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if len(devices) > 0:
        args.devices = devices
        args.device = devices[0]
        args.n_gpu = len(devices)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

     # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    config.num_labels = num_labels

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Task Fine-tuning parameters %s", args)

    if meta_training:
        cuda_magic()
    if args.num_train_epochs > 0:
        global_step, tr_loss, te_loss = train(args, train_dataset, model, tokenizer, meta_training)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if not meta_training:
        if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            print(args.output_dir)
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model # Tae care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments togehter with the trained model
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

            # Load a trained model and vocabulary that you have fine-tund
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=True)
            model.to(args.device)

        # Evaluation
        results = {}
        if args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=True)
            checkpoints = [args.output_dir]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                result = evaluate(args, model, tokenizer, prefix=global_step)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)
        results = results
    else:
        results = te_loss

    print(results)
    return results