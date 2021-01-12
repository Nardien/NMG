# Configuration
# Export Configuration to this file (for clarity)

import argparse

from transformers import (
        BertConfig, BertForMaskedLM, BertTokenizer,
        DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,)

MODEL_CLASSES = {
        "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
        }

def _parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Optional parameters
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--block_size", default=510, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number o/wof training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=-1,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='01')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    # Additional argument
    parser.add_argument("--masking", type=str, default="neural",
                        choices=["random", "neural", "whole", "span", "entity",
                                 "punc", "nopt"],
                        help="Decide whether masking random or neural")
    parser.add_argument("--truncate_pretraining_dataset", type=int, default=-1,
                        help="Decide How many data to use for pretraining")
    parser.add_argument("--mute", action='store_true',
                        help="Mute Logger")
    parser.add_argument("--outer_epoch", type=int, default=10,
                        help="Epoch for training mask generator")

    # Additional argument for task fine-tuning
    parser.add_argument("--task_logging_steps", type=int, default=-1,
                        help="Log every X updates steps for task")
    parser.add_argument("--task_save_steps", type=int, default=-1,
                        help="Save checkpoint evey X updates steps.")
    parser.add_argument("--truncate_task_dataset", type=int, default=-1,
                        help="Decide How many data to use for fine-tuning")
    parser.add_argument("--num_train_task_epochs", type=float, default=1.0,
                        help="Total number of training epochs to fine-tune")
    parser.add_argument("--task_max_steps", type=int, default=-1,
                        help="max step argument for task")
    parser.add_argument("--task", type=str, choices=["glue", "qa"], default="qa",
                        help="Decide which task to use")
    parser.add_argument("--dataset", type=str, choices=["squad", "emrqa", "newsqa"], default="")
    parser.add_argument("--qa_dataset", type=str, choices=["squad", "emrqa", "newsqa"],
                        default="squad", help="In case of QA, indicate which QA dataset going to use")
    parser.add_argument("--glue_dataset", type=str, choices=["chemprot", "imdb"], default="mnli")
    # For Testing
    parser.add_argument("--pytorch_version", type=float, default=1.2)
    parser.add_argument("--mask_learning_rate", type=float, default=5e-3)
    # ================================================================= #
    parser.add_argument("--memory_capacity", type=int, default=200000)
    parser.add_argument("--replay_step", type=int, default=10)
    parser.add_argument("--replay_start", type=int, default=1000)
    parser.add_argument("--replay_batch_size", type=int, default=64)
    # ================================================================= #
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--colorize", action='store_true')
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Checkpoint for testing")
    parser.add_argument("--mask_dir", default="mask/last", type=str,
                        help="Directory for NMG checkpoint")
    parser.add_argument("--log_file", default=None, type=str)
    # ================================================================= #
    parser.add_argument("--repeated_init", action='store_true')
    parser.add_argument("--matching_sample", action='store_true')
    parser.add_argument("--actor_critic", action='store_true')
    parser.add_argument("--fix_task", type=int, default=-1)
    parser.add_argument("--fix_set", action='store_true')
    parser.add_argument("--masking_prob", type=float, default=0.15)
    parser.add_argument("--sampling_strategy", type=str, choices=["R", "V", "A", "Z"],
                        default="A", nargs="+")
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--importance_sampling", action='store_true')
    parser.add_argument("--continual", action='store_true')
    parser.add_argument("--self_play", default="random", type=str,
                        choices=["random", "frozen", "learning"],
                        help="Set Base Agent Type")
    parser.add_argument("--ppo_policy", action='store_true')
    args = parser.parse_args()

    if args.task == "qa":
        args.qa_dataset = args.dataset
    elif args.task in ["glue"]:
        args.glue_dataset = args.dataset

    if not (args.test or args.visualize or args.colorize):
        args = argument_allocation(args)
    return args

def argument_allocation(args):
    # Context Allocation
    if args.dataset == "squad":
        args.train_data_file = "./dataset/SQuAD/squad_train.txt"
    elif args.dataset == "emrqa":
        args.train_data_file = "./dataset/emrQA/emrqa_train.txt"
    elif args.dataset == "newsqa":
        args.train_data_file = "./dataset/NewsQA/newsqa_train.txt"
    else:
        raise NotImplementedError

    if args.model_type == 'distilbert':
        args.config_path = "distilbert-base-uncased"
        args.tokenizer_path = "distilbert-base-uncased"
        args.model_weight_path = "distilbert-base-uncased"
        args.do_lower_case = True
    elif args.model_type == 'bert':
        # Use distilroberta instead of roberta
        model_name = "bert-base-uncased"
        args.tokenizer_path = model_name
        args.config_path = model_name
        args.model_weight_path = model_name
        args.do_lower_case = True
    else:
        raise NotImplementedError
    return args