
import os
import copy
import argparse
import logging
import torch
import numpy as np
import random
import pickle
import time
import sys
import warnings
import shutil

from subprocess import check_call, check_output
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                             TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
# from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
#                                   BertConfig, BertForQuestionAnswering, BertTokenizer,
#                                   DistilBertConfig,
#                                   DistilBertForQuestionAnswering,
#                                   DistilBertTokenizer,
#                                   )

from pytorch_transformers import (WEIGHTS_NAME, WarmupLinearSchedule)

from transformers import (
                    AdamW,
                    BertConfig, BertForQuestionAnswering, BertTokenizer,
                    DistilBertConfig,
                    DistilBertForQuestionAnswering,
                    DistilBertTokenizer,
                    )

from qa_util import (read_squad_examples, convert_examples_to_features,
                     RawResult, write_predictions,
                     RawResultExtended, write_predictions_extended,
                     )


from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad


logger = logging.getLogger(__name__)

PYTHON = sys.executable

MODEL_CLASSES = {
        "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
        }

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def argument_reset(args_origin):
    args = copy.deepcopy(args_origin)
    if args.qa_dataset == "squad":
        args.output_dir = os.path.join(args_origin.output_dir, "SQuAD")
        args.train_file = "./dataset/SQuAD/SQuAD.jsonl"
        args.predict_file = "./dataset/SQuAD/squad_dev.json"
    elif args.qa_dataset == "emrqa":
        args.output_dir = os.path.join(args_origin.output_dir, "emrQA")
        args.train_file = "./dataset/emrQA/emrQA_train.json"
        args.predict_file = "./dataset/emrQA/emrQA_dev.json"
    elif args.qa_dataset == "newsqa":
        args.output_dir = os.path.join(args_origin.output_dir, "newsQA")
        args.train_file = "./dataset/NewsQA/NewsQA.jsonl"
        args.predict_file = "./dataset/NewsQA/NewsQA_dev.jsonl"

    args.model_name_or_path = args_origin.output_dir

    # Specific Hard-Coded Task Argument (Must not be changed)
    args.version_2_with_negative = False
    args.null_score_diff_threshold = 0.0
    args.max_seq_length = 384
    args.doc_stride = 128
    args.max_query_length = 64
    args.n_best_size = 20
    args.max_answer_length = 30
    args.verbose_logging = False
    args.learning_rate = 3e-5

    args.logging_steps = args_origin.task_logging_steps
    args.save_steps = args_origin.task_save_steps
    args.truncate_task_dataset = args_origin.truncate_task_dataset
    args.overwrite_output_dir = True
    args.num_train_epochs = args_origin.num_train_task_epochs
    args.max_steps = args.task_max_steps
    return args


def train(args, train_dataset, model, tokenizer, meta_training):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Add Meta-training Part( Make test set from training set instead of using dev set )
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
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Inner-Loop Fine-tuning", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0],)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2],
                      'start_positions':batch[3],
                      'end_positions':  batch[4]}

            if args.model_type in ["distilbert"]:
                del inputs["token_type_ids"]
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
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer)
                logging_loss = tr_loss

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    te_loss = 0.0
    acc = 0.0
    if meta_training:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        test_step = 0
        # Test for meta-training
        test_sampler = SequentialSampler(test_dataset) if args.local_rank in [-1, 0] else DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size = args.eval_batch_size,)

        # Test
        logger.info(" ***** Running testing for meta-training ***** ")
        logger.info(" Num examples = %d ", len(test_dataset))
        logger.info(" Batch size = %d ", args.train_batch_size)

        all_results = []
        num_samples = 0
        for batch in tqdm(test_dataloader, desc="Inner-Loop Testing"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None if args.model_type == 'xlm' else batch[2],
                          'start_positions':batch[3],
                          'end_positions':  batch[4]}

                if args.model_type in ["distilbert"]:
                    del inputs["token_type_ids"]
                outputs = model(**inputs)
                s_idxes = outputs[1].max(1)[1]
                e_idxes = outputs[2].max(1)[1]
                s_acc = (s_idxes.float() == batch[3].float()).sum()
                e_acc = (e_idxes.float() == batch[4].float()).sum()
                acc += (s_acc.item() + e_acc.item()) / 2

                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()

                te_loss += loss.item()
            test_step += 1
            num_samples += s_idxes.size(0)
        acc = (acc / num_samples) * 100
        te_loss = te_loss / test_step

    return global_step, tr_loss / global_step, acc


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler sampls randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank in [-1, 0] else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size = args.eval_batch_size)

    logger.info(" ***** Running evaluation {} ***** ".format(prefix))
    logger.info("  Num examples = %d  ", len(dataset))
    logger.info("  Batch size = %d  ", args.eval_batch_size)

    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2],
                     }

            if args.model_type in ["distilbert"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]
            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    write_predictions(examples, features, all_results, args.n_best_size,
                    args.max_answer_length, True, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                    args.version_2_with_negative, args.null_score_diff_threshold)
    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)

    return results

def load_examples(args, tokenizer, evaluate=False, output_examples=False, meta_training=False, sampled_keys=None):
    args = argument_reset(args)

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    model_name = args.model_type
    cached_features_file = os.path.join(os.path.dirname(input_file),
            'cached_{}_{}_{}_{}'.format(
        args.qa_dataset,
        'dev' if evaluate else 'train',
        model_name,
        str(args.max_seq_length)))

    print("Load %s" % cached_features_file)

    cached_examples_file = os.path.join(os.path.dirname(input_file), "examples_{}_valid.pkl".format(args.qa_dataset))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)

    elif os.path.exists(cached_features_file) and output_examples and os.path.exists(cached_examples_file):
        logger.info("Loading features and examples from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)
        with open(cached_examples_file, 'rb') as f:
            examples = pickle.load(f)

    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       version_2_with_negative=args.version_2_with_negative)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                )

        if not evaluate:
            max_id = examples[-1].context_id
            tmp_features = {str(i):[] for i in range(max_id+1)}
            for feature in features:
                context_id = feature.context_id
                tmp_features[str(context_id)].append(feature)
            features = tmp_features

        if evaluate and args.local_rank in [-1, 0]:
            logger.info("Dumping evaluation examples to cached file %s", cached_examples_file)
            with open(cached_examples_file, 'wb') as f:
                pickle.dump(examples, f)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as f:
                pickle.dump(features, f)
    if not output_examples:
        examples = None

    return cached_features_file, features, examples

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, meta_training=False, sampled_keys=None):
    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    model_name = args.model_type
    cached_features_file = os.path.join(os.path.dirname(input_file),
            'cached_{}_{}_{}_{}'.format(
        args.qa_dataset,
        'dev' if evaluate else 'train',
        model_name,
        str(args.max_seq_length)))

    print("Load %s" % cached_features_file)

    cached_examples_file = os.path.join(os.path.dirname(input_file), "examples_{}_valid.pkl".format(args.qa_dataset))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)

    elif os.path.exists(cached_features_file) and output_examples and os.path.exists(cached_examples_file):
        logger.info("Loading features and examples from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as f:
            features = pickle.load(f)
        with open(cached_examples_file, 'rb') as f:
            examples = pickle.load(f)

    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       version_2_with_negative=args.version_2_with_negative)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])

        if not evaluate:
            max_id = examples[-1].context_id
            tmp_features = {str(i):[] for i in range(max_id+1)}
            for feature in features:
                context_id = feature.context_id
                tmp_features[str(context_id)].append(feature)
            features = tmp_features

        if evaluate and args.local_rank in [-1, 0]:
            logger.info("Dumping evaluation examples to cached file %s", cached_examples_file)
            with open(cached_examples_file, 'wb') as f:
                pickle.dump(examples, f)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)
            with open(cached_features_file, 'wb') as f:
                pickle.dump(features, f)

    if not evaluate and args.matching_sample:
        train_features = []
        for key in sampled_keys:
            feature = features[key]
            train_features += feature
        train_feature_length = len(train_features) if args.truncate_task_dataset < 0 \
                               else min(args.truncate_task_dataset, len(train_features))
        extracted_features = random.sample(features.items(), len(features))

        if args.fix_task > 0:
            print("Load Fixed Valid Task...")
            feature_length = args.fix_task
            valid_features_file = cached_features_file + "_valid_" + str(feature_length)
            if os.path.exists(valid_features_file):
                with open(valid_features_file, 'rb') as f:
                    test_features = pickle.load(f)
            else:
                sub_features = []
                for feature in features.items():
                    sub_features += feature[1]
                sub_len = len(sub_features)
                feature_length = sub_len if args.truncate_task_dataset < 0 \
                                 else min(sub_len, feature_length)
                test_features = random.sample(sub_features,
                                              feature_length)
                with open(valid_features_file, 'wb') as f:
                    pickle.dump(test_features, f)

        else:
            test_features = []
            count = 0
            for feature in extracted_features:
                if count >= train_feature_length:
                    break
                test_features += feature[1]
                count += len(feature[1])
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
        for idx, feature in features.items():
            all_features += feature
        features = all_features

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if not meta_training:
        # Convert to Tensors and build dataset
        if args.truncate_task_dataset > 0 and not evaluate:
            features = features[:args.truncate_task_dataset]
        print('Extracted features --> %s' % len(features))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        if evaluate:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_example_index, all_cls_index, all_p_mask)
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
    else:
        if not evaluate and not args.matching_sample:
            if args.truncate_task_dataset > 0 and not evaluate:
                random.shuffle(features)
                features = features[:args.truncate_task_dataset]

            train_len = int(0.8 * len(features))
            test_features = features[train_len:]
            train_features = features[:train_len]
            print('Extracted train features --> %s' % len(train_features))
            print('Extracted test features --> %s' % len(test_features))

        dataset = []

        for features in [train_features, test_features]:
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset_ = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
            dataset.append(dataset_)


    if output_examples:
        return dataset, examples, features
    return dataset

# CUDA_MAGIC -> avoid cuda gpu exploration
def cuda_magic():
    start = time.time()
    tmp = "./dataset/SQuAD/cached_lm_510_squad_train.txt_bert"
    with open(tmp, 'rb') as f:
        _tmp = pickle.load(f)
    del _tmp
    print("Elapsed Time for CUDA Magic... {} sec".format(time.time() - start))

def run_qa(args, train_dataset, meta_training, reset_env=True, devices=[], sampled_keys=None):
    # Reallocate argument for Task
    args = argument_reset(args)

    if not args.overwrite_output_dir and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Note taht if local_rank is not -1...
    if len(devices) > 0:
        args.devices = devices
        args.device = devices[0]
        args.n_gpu = len(devices)
    logger.warning("Process %d: Enter Task worker with device %s", args.local_rank, args.device)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path)

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
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
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
            tokenizer = tokenizer_class.from_pretrained(args.output_dir,
                        do_lower_case=args.do_lower_case)
            model.to(args.device)

        # Evaluation
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        results = {}
        if args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(args.output_dir,
                        do_lower_case=args.do_lower_case)
            checkpoints = [args.output_dir]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                # Evaluate
                result = evaluate(args, model, tokenizer, prefix=global_step)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                results.update(result)

            if args.local_rank != -1:
                torch.distributed.barrier()

        results = results
    else:
        results = te_loss

    print(results)
    return results

