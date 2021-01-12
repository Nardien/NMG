import random
from pretrain_util import load_and_cache_examples
import torch
import numpy as np
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm

class NMGDataset:
    def __init__(self, args, tokenizer, evaluate, meta_training):
        # Load Context
        self.context_dataset = load_and_cache_examples(args, tokenizer,
                evaluate=evaluate,
                meta_training=meta_training)

        if args.task == "qa":
            from qa_worker import load_examples
        else:
            raise NotImplementedError

        # Load Task Dataset
        # task_dataset = (cache, features, examples)
        self.task_dataset = load_examples(args, tokenizer,
                evaluate=evaluate,
                meta_training=meta_training)

        if meta_training:
            # Split for Train / Valid
            self.context_dataset.split()
            cache, features, examples = self.task_dataset
            self.task_valid_dataset = self.sample_task(args, features,
                                        self.context_dataset.valid_sampled_keys,
                                        validate=True)

    def sample(self, args):
        # Sample Context
        self.context_dataset.sample(args.truncate_pretraining_dataset)
        print("Sampled Context Length --> {}".format(len(self.context_dataset)))

        # Sample Task
        cache, features, examples = self.task_dataset
        sampled_task_dataset = self.sample_task(args, features,
                                self.context_dataset.train_sampled_keys)
        return self.context_dataset, \
            [sampled_task_dataset, self.task_valid_dataset]


    def sample_task(self, args, features, sampled_keys, validate=False):
        sampled_features = []
        for key in sampled_keys:
            feature = features[key]
            sampled_features += feature

        if not validate:
            sampled_feature_length = len(sampled_features) if args.truncate_task_dataset < 0 \
                    else min(args.truncate_task_dataset, len(sampled_features))
        else:
            sampled_feature_length = len(sampled_features) if args.fix_task < 0 \
                    else min(args.fix_task, len(sampled_features))

        sampled_features = random.sample(sampled_features,
                                         sampled_feature_length)


        print("Sampled Task Length --> {}".format(len(sampled_features)))

        features = sampled_features
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if args.task == "qa":
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
        elif args.task == "glue":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask,
                                    all_segment_ids, all_label_ids)
        else:
            raise NotImplementedError
        return dataset

    # For Meta-Testing
    def get(self, args):
        # Sample context?
        if args.truncate_pretraining_dataset > 0:
            self.context_dataset.sample(args.truncate_pretraining_dataset)
        print("Context Length --> {}".format(len(self.context_dataset)))

        cache, features_dict, examples = self.task_dataset
        features = []
        for key in features_dict.keys():
            features += features_dict[key]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if args.task == "qa":
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)
        elif args.task == "glue":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask,
                                    all_segment_ids, all_label_ids)
        else:
            raise NotImplementedError
        print("Task Length --> {}".format(len(features)))

        return self.context_dataset, dataset