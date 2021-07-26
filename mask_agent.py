# This code is for Neural mask generator
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
import random
import logging
import copy
import json
from tqdm import tqdm
from transformers import AdamW
from tensorboardX import SummaryWriter
from nltk.corpus import stopwords
import numpy as np
import string
import spacy
import pandas as pd
import time
from collections import Counter
logger = logging.getLogger(__name__)

class MemoryEntry(object):
    def __init__(self, state, action, log_prob=None, value=None, feature=None,
            reward=None, z=True, sub_p=1, raw_input=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.log_prob = log_prob
        self.feature = feature
        self.value = value
        self.z = z
        self.label = None
        self.sub_p = sub_p
        self.raw_input = raw_input

    def update_reward(self, reward):
        if self.reward is not None:
            self.reward *= reward
        else:
            self.reward = reward

    def update_label(self, label):
        self.label = label

class StateContainer(object):
    def __init__(self, data):
        # Such kind of heap memory
        self.data = data
        self.counter = 0

    def counter_add(self):
        self.counter += 1

    def counter_minus(self):
        self.counter -= 1

EPS = 1e-15
class MaskGenerator(object):
    def __init__(self, args, model_base, config, tokenizer, training=True, base=False):
        self.args = args
        self.pytorch_version = args.pytorch_version
        self.masking_type = args.masking
        # self.gap_acc_reward = args.gap_acc_reward

        self.tokenizer = tokenizer
        # Construct Stopword list
        stop_words=[]
        for c in string.punctuation:
            stop_words.append(c)
        self.stop_words = list(set(stop_words))
        self.stop_words.append("[CLS]")
        self.stop_words.append("[SEP]")
        self.stop_words.append("[UNK]")
        self.log_file = os.path.join(args.output_dir, "training_logs.txt")

        if self.masking_type == 'entity':
            self.nlp = spacy.load("en_core_web_sm")
        # Indicate that whether this mask generator is base or not
        self.base = base
        self.training = training

        # Set Mask Generating model inside the mask generator
        self.model = model_base

        # Lock the bert parameters
        if self.model is not None and not args.continual:
            for p in self.model.bert.parameters():
                p.requires_grad = False

        if self.model is not None and self.training:
            # Set Optimizer for training neural mask generator
            no_decay = ['bias', 'LayerNorm.weight']
            learnable_weight = []
            learnable_bias = []
            for n, p in self.model.named_parameters():
                if 'bert' not in n and 'weight' in n:
                    learnable_weight.append(n)
                if 'bert' not in n and 'bias' in n:
                    learnable_bias.append(n)
            optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if n in learnable_weight], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if n in learnable_bias],'weight_decay': 0.0}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.mask_learning_rate, eps=args.adam_epsilon)

        self.adv_rewards = []
        self.adv_losses = []
        self.rewards = []
        self.reward_limit = 10

        # History for RL training
        self.initialize_history()
        # Accuracy is shared accross episode
        self.former_accuracy = None
        self.acc_history = []
        self.base_acc_history = []
        self.rnd_acc_history = []
        self.best_diff = -100
        self.best_improve_indicator = 0
        self.improve_indicator = 0
        self.rand_self_indicator = 0
        self.regret = 0

        self.replay_memory = []
        self.state_containers = []

        self.global_step = 0

        self.episode = 0
        # Tensorboard writer for mask generator
        if self.training:
            self.tb_writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

        # Initialize mask checkpoint directory

        if self.base and args.self_play == "learning":
            self.mask_base_dir = os.path.join(args.output_dir, "mask_base")
        else:
            self.mask_base_dir = os.path.join(args.output_dir, "mask")

        if not os.path.exists(self.mask_base_dir) and args.local_rank in [-1, 0]:
            os.makedirs(self.mask_base_dir)

        self.mask_last_dir = os.path.join(self.mask_base_dir, "last")
        if not os.path.exists(self.mask_last_dir) and args.local_rank in [-1, 0]:
            os.makedirs(self.mask_last_dir)

        self.checkpoint = os.path.join(self.mask_base_dir, "checkpoint")
        if not os.path.exists(self.checkpoint) and args.local_rank in [-1, 0]:
            os.makedirs(self.checkpoint)

        self.profile_dir = os.path.join(args.output_dir, "replay_profile_dir")
        os.makedirs(self.profile_dir, exist_ok=True)

    def save(self, episode):
        if "neural" not in self.masking_type: return

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        logger.info("Saving last mask model checkpoint to %s", self.mask_last_dir)
        model_to_save.save_pretrained(self.mask_last_dir)

        if episode % 50 == 0:
            mask_tmp_dir = os.path.join(self.mask_base_dir, "{}".format(episode))
            if not os.path.exists(mask_tmp_dir) and self.args.local_rank in [-1, 0]:
                os.makedirs(mask_tmp_dir)
            logger.info("Saving %d-th mask model checkpoint to %s" % (episode, mask_tmp_dir))
            model_to_save.save_pretrained(mask_tmp_dir)

        if episode % 10 == 0:
            model_to_save.save_pretrained(self.checkpoint)
            torch.save({
                        'optim': self.optimizer.state_dict(),
                        'episode': self.episode,
                       },
                       os.path.join(self.checkpoint, "last.ckpt"))

    def initialize_history(self):
        self.masked_log_probs = []
        self.log_probs = []
        while len(self.rewards) > self.reward_limit:
            self.rewards.pop(0)
        self.non_mask_ratios = []
        self.entropy = []

        self.action_history = []

        self.tmp_replay_memory = []

    def set_masking_type(self, masking_type):
        self.masking_type = masking_type

    def mask(self, inputs, tokenizer, args, visualize=False, model=None):
        masking_type = self.masking_type
        if masking_type == "random":
            outputs = self.random_mask_tokens(inputs, tokenizer, args, visualize)
        elif masking_type  == "neural":
            outputs = self.neural_mask_tokens(inputs, tokenizer, args, visualize, model)
        elif masking_type == "whole":
            outputs = self.whole_random_mask_tokens(inputs, tokenizer, args, visualize)
        elif masking_type == "span":
            outputs = self.span_random_mask_tokens(inputs, tokenizer, args, visualize)
        elif masking_type == "entity":
            outputs = self.entity_random_mask_tokens(inputs, tokenizer, args, visualize)
        elif masking_type == "punc":
            outputs = self.punc_random_mask_tokens(inputs, tokenizer, args, visualize)

        return outputs

    def neural_mask_tokens(self, inputs, tokenizer, args, visualize=False, model=None):
        """ Prepare masked tokens based on neural mask generator. """
        if args.device2 is not None: device = args.device2
        else: device = args.device

        self.device = device

        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)

        labels = inputs.clone()
        inputs = inputs.to(device)
        raw_inputs = inputs.clone()
        input_mask = ~inputs.eq(args.pad_token)

        masked_inputs = raw_inputs

        _input_mask = input_mask
        seq_length = inputs.shape[-1]

        self.model.eval()

        if args.continual:
            original_device = next(model.parameters()).device
            # Temporarily send to extract feature
            model.to(device)
            with torch.no_grad():
                model.eval()
                if hasattr(model, 'bert'):
                    language_model = model.bert
                elif hasattr(model, 'distilbert'):
                    language_model = model.distilbert
                else:
                    raise NotImplementedError
                outputs = language_model(inputs,
                          attention_mask=input_mask)
            inputs = outputs[0]
            model.to(original_device)


        with torch.no_grad():
            outputs = self.model(inputs,
                    attention_mask=input_mask, visualize=visualize,
                    args=args)

        logit, value = outputs[0], outputs[1]
        logit = logit * input_mask.to(torch.float)
        logit[logit == 0] = -float('inf')

        prob = F.softmax(logit, 1)
        log_prob = F.log_softmax(logit, 1)

        num_samples = [max(int(i.sum().item() * args.masking_prob), 1) for i in input_mask]
        entropy = []

        for i, num in enumerate(input_mask):
            non_mask_num = int(num.sum().item())
            entropy.append(-(log_prob[i][:non_mask_num] * prob[i][:non_mask_num]).sum().item())

        batch_log_probs = []
        for i in range(len(num_samples)):
            nmn = int(input_mask[i].sum().item())
            _prob = prob[i][:nmn] + EPS

            if self.training:
                normalize_counter = Counter()
                normalize_counter.update(raw_inputs[i][:nmn].tolist())
                normalize_factor = \
                        [1 / np.sqrt(normalize_counter[w]) for w in raw_inputs[i][:nmn].tolist()]

            if self.training and (args.self_play != "frozen" or not self.base):
                action = _prob.multinomial(num_samples=num_samples[i])
            else:
                action = _prob.topk(k=num_samples[i], dim=0)[1]

            self.action_history.append(action.tolist())

            if self.training and args.save_history:
                if args.self_play == "learning" or (args.self_play in ["frozen", "random"] and not self.base):
                    replay_memory_same_context = []
                    for a in action:
                        feature = None
                        v = value[i]
                        sub_p = normalize_factor[a]
                        replay_memory_same_context.append(MemoryEntry(inputs[i].cpu(),
                                                                      a.cpu(),
                                                                      log_prob[i][a].detach().cpu(),
                                                                      v.detach().cpu(),
                                                                      feature=feature,
                                                                      sub_p=sub_p,
                                                                      raw_input=raw_inputs[i].cpu(),
                                                                      ))
                    self.tmp_replay_memory.append(replay_memory_same_context)


                elif not self.base:
                    for a in action:
                        feature = None
                        v = value[i]
                        self.tmp_replay_memory.append(MemoryEntry(inputs[i].cpu(),
                                                                  a.cpu(),
                                                                  log_prob[i][a].detach().cpu(),
                                                                  v.detach().cpu(),
                                                                  feature=feature))

            if self.training and False:
                log_prob_ = log_prob[i].gather(0, action)
                batch_log_probs.append(log_prob_)

            masked_indices = torch.zeros(labels[i].shape).to(device).scatter_(0, action, 1).bool()
            labels[i][~masked_indices] = -100

            _input_mask = (~masked_inputs[i].eq(args.pad_token)).to(torch.float)
            indices_replaced = torch.bernoulli(_input_mask * 0.8).bool() & masked_indices
            masked_inputs[i][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

            indices_random = torch.bernoulli(_input_mask * 0.5).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(tokenizer), labels[i].shape, dtype=torch.long).cuda(device)
            masked_inputs[i][indices_random] = random_words[indices_random]

        if self.training:
            if False:
                self.log_probs.append(torch.cat(batch_log_probs))
            self.entropy.append(sum(entropy) / len(entropy))

        if visualize:
            non_mask_num = int(input_mask.sum().item())
            mask_prob = prob[0][:non_mask_num].cpu().numpy()
            return masked_inputs, labels, mask_prob, action
        return masked_inputs, labels

    def random_mask_tokens(self, inputs, tokenizer, args, visualize=False):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)

        labels = inputs.clone()
        puncs = [tokenizer.convert_tokens_to_ids(p) for p in list(string.punctuation)]

        input_mask = (~inputs.eq(0)).to(torch.float)
        masking_prob = args.masking_prob
        # masked position = 1 or 0
        # Consider padding
        masked_indices = torch.bernoulli(input_mask * masking_prob).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(input_mask * 0.8).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = torch.bernoulli(input_mask * 0.5).bool() & masked_indices & ~indices_replaced
        
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        if self.base:
            for i in range(masked_indices.shape[0]):
                action = masked_indices[i].nonzero().view(-1).tolist()
                self.action_history.append(action)

        return inputs, labels

    def whole_random_mask_tokens(self, inputs, tokenizer, args, visualize=False):
        """ Prepare masked tokens for masked language modeling with whole word masking. """
        if args.device2 is not None: device = args.device2
        else: device = args.device
        labels = []
        new_inputs = []
        # Doing operation per batch
        for input_raw in inputs:
            label_raw = input_raw.clone()

            input_mask = (~input_raw.eq(0))
            tokens = tokenizer.convert_ids_to_tokens(input_raw.tolist())

            cand_indexes = []
            # Aggregate Subtokens to one token
            for (i, token) in enumerate(tokens):
                if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                    continue
                if args.stopword_masking and token in self.stop_words:
                    continue
                if len(cand_indexes) >= 1 and token.startswith("##") and not args.stopword_masking:
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])

            random.shuffle(cand_indexes)

            # Ignore padding
            num_to_predict = max(1, int(round(input_mask.sum().item() * args.masking_prob)))

            masked_lms = []
            covered_indexes = set()
            for index_set in cand_indexes:
                if len(masked_lms) >= num_to_predict:
                    break

                if len(masked_lms) + len(index_set) > num_to_predict:
                    continue

                is_any_index_covered = False
                for index in index_set:
                    if index in covered_indexes:
                        is_any_index_covered = True
                        break

                if is_any_index_covered:
                    continue

                for index in index_set:
                    covered_indexes.add(index)

                    masked_token = None

                    # 80% of time, replace with [MASK]
                    if random.random() < 0.8:
                        masked_token = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                    else:
                        if random.random() < 0.5:
                            masked_token = tokenizer.convert_tokens_to_ids(tokens[index])
                        else:
                            masked_token = random.randint(0, len(tokenizer) - 1)
                    tokens[index] = masked_token

                    masked_lms.append((index, tokens[index]))

            assert len(masked_lms) <= num_to_predict

            masked_lms = sorted(masked_lms, key=lambda x: x[0])

            masked_lms_pointer = 0
            for i in range(input_raw.shape[0]):
                if masked_lms_pointer < len(masked_lms) and i == masked_lms[masked_lms_pointer][0]:
                    input_raw[i] = masked_lms[masked_lms_pointer][1]
                    masked_lms_pointer += 1
                else:
                    label_raw[i] = -100

            new_inputs.append(input_raw)
            labels.append(label_raw)

        labels = torch.stack(labels, 0)
        inputs = torch.stack(new_inputs, 0)

        return inputs, labels

    def span_random_mask_tokens(self, inputs, tokenizer, args, visualize=False):
        """ Prepare masked tokens for masked language modeling with whole word masking. """
        if args.device2 is not None: device = args.device2
        else: device = args.device
        labels = []
        new_inputs = []
        # Doing operation per batch
        for input_raw in inputs:
            label_raw = input_raw.clone()

            # Geometric distribution for sampling span length
            geometric = torch.distributions.geometric.Geometric(0.2)

            input_mask = (~input_raw.eq(0))

            tokens = tokenizer.convert_ids_to_tokens(input_raw.tolist())

            cand_indexes = []
            # Aggregate Subtokens to one token
            for (i, token) in enumerate(tokens):
                if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                    continue

                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)

                else:
                    cand_indexes.append([i])

            num_to_predict = max(1, int(round(input_mask.sum().item() * args.masking_prob)))

            masked_lms = []
            covered_indexes = set()

            tolerance = 0

            while len(masked_lms) < num_to_predict:
                tolerance += 1
                # Breaking infinite loof (in case of cannot finding exact length for num_to_predict...)
                if tolerance > 1000000:
                    print("  I CAN'T TOLERATE ANY MORE!!!  ")
                    break

                # Randomly pick the starting cand indexes
                start = int(random.random() * len(cand_indexes))

                # If start index is already masked, continue
                if cand_indexes[start][0] in covered_indexes:
                    continue

                # Sample the length of random spans
                l = geometric.sample().item() + 1
                if l > 10: l = 10 # Clipping length following the spanBERT paper

                l = int(l)
                masked_token_policy = None
                # Decide How to masking tokens in same span
                if random.random() < 0.8:
                    masked_token_policy = "mask"
                else:
                    if random.random() < 0.5:
                        masked_token_policy = "random"
                    else:
                        masked_token_policy = "original"
                # Q: What if the index exceeds the maximum length..?
                sampled_span = cand_indexes[start:start + l]
                index_set = []
                for idx in sampled_span:
                    index_set += idx
                if len(masked_lms) + len(index_set) > num_to_predict:
                    continue
                is_any_index_covered = False
                for index in index_set:
                    if index in covered_indexes:
                        is_any_index_covered = True
                        break
                if is_any_index_covered:
                    continue
                tolerance = 0
                for index in index_set:
                    covered_indexes.add(index)
                    masked_token = None
                    if masked_token_policy == "mask":
                        masked_token = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                    elif masked_token_policy == "original":
                        masked_token = tokenizer.convert_tokens_to_ids(tokens[index])
                    elif masked_token_policy == "random":
                        masked_token = random.randint(0, len(tokenizer) - 1)
                    else:
                        assert False
                    masked_lms.append((index, masked_token))

            assert len(masked_lms) <= num_to_predict

            masked_lms = sorted(masked_lms, key=lambda x: x[0])
            masked_lms_pointer = 0
            for i in range(input_raw.shape[0]):
                if masked_lms_pointer < len(masked_lms) and i == masked_lms[masked_lms_pointer][0]:
                    input_raw[i] = masked_lms[masked_lms_pointer][1]
                    masked_lms_pointer += 1
                else:
                    label_raw[i] = -100
            new_inputs.append(input_raw)
            labels.append(label_raw)

        labels = torch.stack(labels, 0)
        inputs = torch.stack(new_inputs, 0)

        if visualize:
            action = []
            for idx, label in enumerate(labels.squeeze(0)):
                if label.item() != -1:
                    action.append(idx)
            return inputs, labels, action

        return inputs, labels

    def entity_random_mask_tokens(self, inputs, tokenizer, args, visualize=False):
        """ Prepare masked tokens for masked language modeling with whole word masking. """
        if args.device2 is not None: device = args.device2
        else: device = args.device
        labels = []
        new_inputs = []
        # Doing operation per batch
        for input_raw in inputs:
            label_raw = input_raw.clone()

            input_mask = (~input_raw.eq(0))
            tokens = tokenizer.convert_ids_to_tokens(input_raw.tolist())

            cand_tokens = []
            for token in tokens[:input_mask.sum().item()]:
                if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                    continue
                if len(cand_tokens) >= 1 and token.startswith("##"):
                    cand_tokens[-1].append(token)
                else:
                    cand_tokens.append([token])

            char_to_idx = []
            # It matches to cand_indexes
            sentence = ''
            for (i, t) in enumerate(cand_tokens):
                word = ' '.join(t).replace(" ##", "").replace("##","")
                sentence += word
                for _ in range(len(word)):
                    char_to_idx.append(i)
                sentence += ' '
                char_to_idx.append(i)

            cand_indexes = []
            # Aggregate Subtokens to one token
            for (i, token) in enumerate(tokens):
                if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                    continue
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])

            doc = self.nlp(sentence)
            entity_idx = []
            for ent in doc.ents:
                start_idx = char_to_idx[ent.start_char]
                end_idx = char_to_idx[ent.end_char]
                for i in range(start_idx, end_idx+1):
                    entity_idx.append(i)

            entity_cand_indexes = []
            non_cand_indexes = []
            for num, idxes in enumerate(cand_indexes):
                if num in entity_idx:
                    entity_cand_indexes.append(idxes)
                else:
                    non_cand_indexes.append(idxes)

            random.shuffle(entity_cand_indexes)
            random.shuffle(non_cand_indexes)
            cand_indexes = entity_cand_indexes + non_cand_indexes

            # Ignore padding
            num_to_predict = max(1, int(round(input_mask.sum().item() * args.masking_prob)))

            masked_lms = []
            covered_indexes = set()
            for index_set in cand_indexes:
                if len(masked_lms) >= num_to_predict:
                    break

                if len(masked_lms) + len(index_set) > num_to_predict:
                    continue

                is_any_index_covered = False
                for index in index_set:
                    if index in covered_indexes:
                        is_any_index_covered = True
                        break

                if is_any_index_covered:
                    continue

                for index in index_set:
                    covered_indexes.add(index)

                    masked_token = None

                    # 80% of time, replace with [MASK]
                    if random.random() < 0.8:
                        masked_token = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                    else:
                        if random.random() < 0.5:
                            masked_token = tokenizer.convert_tokens_to_ids(tokens[index])
                        else:
                            masked_token = random.randint(0, len(tokenizer) - 1)
                    tokens[index] = masked_token

                    masked_lms.append((index, tokens[index]))

            assert len(masked_lms) <= num_to_predict

            masked_lms = sorted(masked_lms, key=lambda x: x[0])

            masked_lms_pointer = 0
            for i in range(input_raw.shape[0]):
                if masked_lms_pointer < len(masked_lms) and i == masked_lms[masked_lms_pointer][0]:
                    input_raw[i] = masked_lms[masked_lms_pointer][1]
                    masked_lms_pointer += 1
                else:
                    label_raw[i] = -1

            new_inputs.append(input_raw)
            labels.append(label_raw)

        labels = torch.stack(labels, 0)
        inputs = torch.stack(new_inputs, 0)

        return inputs, labels

    def punc_random_mask_tokens(self, inputs, tokenizer, args, visualize=False):
        if args.device2 is not None: device = args.device2
        else: device = args.device

        labels = []
        new_inputs = []

        puncs = list(string.punctuation)

        for input_raw in inputs:
            label_raw = input_raw.clone()
            input_mask = (~input_raw.eq(0))
            tokens = tokenizer.convert_ids_to_tokens(input_raw.tolist())

            num_to_predict = max(1, int(round(input_mask.sum().item() * args.masking_prob)))

            punc_index = []
            non_punc_index = []
            for (i, t) in enumerate(tokens):
                if t in puncs:
                    punc_index.append(i)
                else:
                    non_punc_index.append(i)
            random.shuffle(punc_index)
            random.shuffle(non_punc_index)
            cand_indexes = punc_index + non_punc_index

            masked_lms = []
            covered_indexes = set()
            for index in cand_indexes:
                if len(masked_lms) >= num_to_predict:
                    break

                is_any_index_covered = False
                if index in covered_indexes:
                    is_any_index_covered = True
                    break

                if is_any_index_covered:
                    continue

                covered_indexes.add(index)
                masked_token = None

                if random.random() < 0.8:
                    masked_token = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                else:
                    if random.random() < 0.5:
                        masked_token = tokenizer.convert_tokens_to_ids(tokens[index])
                    else:
                        masked_token = random.randint(0, len(tokenizer) - 1)
                tokens[index] = masked_token
                masked_lms.append((index, tokens[index]))

            assert len(masked_lms) <= num_to_predict

            masked_lms = sorted(masked_lms, key=lambda x: x[0])
            masked_lms_pointer = 0

            for i in range(input_raw.shape[0]):
                if masked_lms_pointer < len(masked_lms) and i == masked_lms[masked_lms_pointer][0]:
                    input_raw[i] = masked_lms[masked_lms_pointer][1]
                    masked_lms_pointer += 1
                else:
                    label_raw[i] = -1

            new_inputs.append(input_raw)
            labels.append(label_raw)

        labels = torch.stack(labels, 0)
        inputs = torch.stack(new_inputs, 0)

        return inputs, labels

    # op = opponent, pl = player
    def append_reward_selfplay(self, op_acc, pl_acc, op_history,
            pl_history=None, rnd_acc=None, rnd_history=None):
        if "neural" not in self.masking_type:
            return

        pl_history = self.action_history

        reward = pl_acc - op_acc

        if reward > 0:
            reward = 1
            self.improve_indicator += 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1
            self.improve_indicator -= 1
            self.regret += 1

        if rnd_acc is not None:
            rnd_reward = pl_acc - rnd_acc
            rnd_reward = np.sign(rnd_reward)
        else:
            rnd_reward = None

        assert len(op_history) == len(pl_history)
        if rnd_history is not None:
            assert len(rnd_history) == len(pl_history)

        tmp_replay_memory = []
        for i, transitions in enumerate(self.tmp_replay_memory):
            op_actions = op_history[i]
            pl_actions = pl_history[i]
            rnd_actions = rnd_history[i] if rnd_history is not None else pl_actions

            pl_actions_disjoint_op = list(set(pl_actions) - set(op_actions))
            pl_actions_disjoint_rnd = list(set(pl_actions) - set(rnd_actions))

            base_data = None
            for j, entry in enumerate(transitions):
                if entry.action in pl_actions_disjoint_op and entry.action in pl_actions_disjoint_rnd:
                    # Take minimum reward for both disjoint circumstances
                    entry.update_reward(min(reward, rnd_reward) if rnd_history is not None else reward)

                elif entry.action in pl_actions_disjoint_op and entry.action not in pl_actions_disjoint_rnd:
                    entry.update_reward(reward)

                elif entry.action not in pl_actions_disjoint_op and entry.action in pl_actions_disjoint_rnd:
                    entry.update_reward(rnd_reward)

                else:
                    # Joint action - do not learn
                    continue

                if base_data is None:
                    assert entry.state.__class__.__name__ == "Tensor"
                    base_data = StateContainer(entry.state)
                    self.state_containers.append(base_data)

                entry.state = base_data
                base_data.counter_add()

                entry.z = 1
                tmp_replay_memory.append(entry)

        self.replay_memory += tmp_replay_memory

        start = time.time()
        while len(self.replay_memory) > self.args.memory_capacity:
            pop_entry = self.replay_memory.pop(0)
            pop_entry.state.counter_minus()

        # Clean memory with no reference
        delete_count = 0
        for idx, sc in enumerate(self.state_containers):
            if sc.counter == 0:
                self.state_containers.pop(idx)
                delete_count += 1

        while len(self.state_containers) > 5000:
            pop_entry = self.replay_memory.pop(0)
            pop_entry.state.counter_minus()

            sc = self.state_containers[0]
            if sc.counter == 0:
                self.state_containers.pop(0)
                delete_count += 1

        print("Elapsed Time for Pop memory: {}".format(time.time() - start))
        print("Memory len: {}".format(len(self.replay_memory)))

        print("Delete {} states among {}".format(delete_count,
            len(self.state_containers)))

        if not self.base:
            if rnd_reward is not None:
                self.rewards.append(rnd_reward)
            else:
                self.rewards.append(reward)
            self.acc_history.append(pl_acc)
            self.base_acc_history.append(op_acc)

        if rnd_acc is not None and not self.base:
            self.rnd_acc_history.append(rnd_acc)
            if rnd_acc > pl_acc:
                self.rand_self_indicator -= 1
            elif rnd_acc < pl_acc:
                self.rand_self_indicator += 1


    def train_replay(self, args):
        if "neural" not in self.masking_type:
            return

        if args.device2 is not None: device = args.device2
        else: device = args.device

        if len(self.replay_memory) < args.replay_start:
            if not self.base: self.write_logs(rewards=self.rewards)
            return

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        self.model.train()
        self.model.zero_grad()
        for _ in tqdm(range(args.replay_step), desc="ReinforcementLearning", position=0):
            if args.actor_critic:
                r = np.array([e.reward for e in self.replay_memory])
                V = np.array([e.value for e in self.replay_memory])
                _sampling_p = []
                if "R" in args.sampling_strategy:
                    _sampling_p.append(np.exp(r) / np.sum(np.exp(r)))
                if "A" in args.sampling_strategy:
                    _sampling_p.append(np.exp(abs(r-V)) / np.sum(np.exp(abs(r-V))))
                if "V" in args.sampling_strategy:
                    _sampling_p.append(np.exp(V) / np.sum(np.exp(V)))
                # assert len(_sampling_p) > 0

                if len(_sampling_p) > 0:
                    sampling_p = sum(_sampling_p) / len(_sampling_p)
                else:
                    sampling_p = np.array([1 for e in self.replay_memory])
                sampling_p = sampling_p / sum(sampling_p)

                sub_p = np.array([e.sub_p for e in self.replay_memory])
                sampling_p *= sub_p
                sampling_p = sampling_p / sum(sampling_p)
                sampled_batch = np.random.choice(self.replay_memory,
                                                 args.replay_batch_size,
                                                 replace=False,
                                                 p=sampling_p)
            else:
                r = np.array([e.reward for e in self.replay_memory])
                sampling_p = np.exp(r) / np.sum(np.exp(r))
                sampled_batch = np.random.choice(self.replay_memory,
                                                 args.replay_batch_size,
                                                 replace=False,
                                                 p=sampling_p)
            inputs = torch.stack([e.state.data for e in sampled_batch]).to(device)
            action = torch.stack([e.action for e in sampled_batch]).view(-1, 1).to(device)
            reward = torch.tensor([e.reward for e in sampled_batch],
                                  device=device, dtype=torch.float)
            z = torch.tensor([e.z for e in sampled_batch], device=device, dtype=torch.float)

            if args.continual:
                raw_inputs = torch.stack([e.raw_input for e in sampled_batch]).to(device)
                input_mask = (~raw_inputs.eq(args.pad_token)).to(torch.float)
            else:
                input_mask = (~inputs.eq(args.pad_token)).to(torch.float)
            seq_length = inputs.shape[1]
            outputs = self.model(inputs,
                                 attention_mask=input_mask,
                                 args=args)
            if args.actor_critic:
                logit, value = outputs[0], outputs[1]
            else:
                logit = outputs[0]
            logit = logit * input_mask
            logit[logit == 0] = -float('inf')

            prob = F.softmax(logit, 1)
            if 0 in [p for p in prob.sum(-1)]:
                log_prob = torch.log(prob + EPS)
            else:
                log_prob = F.log_softmax(logit, 1)

            action_log_prob = log_prob.gather(1, action).view(-1)

            # Update value
            for batch_idx, entry in enumerate(sampled_batch):
                entry.value = value[batch_idx].detach().cpu()
                
            if args.actor_critic:
                adv = reward * z - value.detach()
            else:
                adv = reward * z

            if args.importance_sampling:
                old_log_prob = torch.stack([e.log_prob for e in sampled_batch]).to(device)
                ratio = torch.exp(action_log_prob - old_log_prob)
                if args.ppo_policy:
                    policy_loss = -(adv * ratio).mean()
                else:
                    policy_loss = -(adv * ratio.detach() * action_log_prob).mean(0)
            else:
                policy_loss = -(adv * action_log_prob).mean(0)

            # Add Entropy Regularizer
            entropy_regularizer = 0
            for i, num in enumerate(input_mask):
                non_mask_num = int(num.sum().item())
                _entropy = (-(log_prob[i][:non_mask_num] * prob[i][:non_mask_num]).mean())
                entropy_regularizer += _entropy

            if not args.actor_critic:
                entropy_regularizer = 0

            value_loss = 0
            if args.actor_critic:
                value_loss = 0.5 * (reward - value).pow(2).mean()
                total_value_loss += value_loss.item()

            loss = 0.5 * value_loss + policy_loss - (args.entropy_coeff * entropy_regularizer)
            # loss = 0.5 * value_loss + policy_loss - (0.01 * entropy_regularizer)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            self.optimizer.step()
            self.model.zero_grad()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()

        tqdm.write(str(total_loss))
        tqdm.write(str(total_policy_loss))
        tqdm.write(str(total_value_loss))
        if not self.base:
            self.write_logs(loss=total_loss / args.replay_step,
                            rewards=self.rewards,
                            policy_loss=total_policy_loss / args.replay_step,
                            value_loss=total_value_loss / args.replay_step)

    def write_logs(self, policy_loss=None, loss=None, rewards=None, value_loss=None):
        # Train = End of the episode
        if len(self.acc_history) > 0:
            self.tb_writer.add_scalar('accuracy', self.acc_history[-1], self.episode)
        if len(self.base_acc_history) > 0:
            self.tb_writer.add_scalar('base_accuracy', self.base_acc_history[-1], self.episode)
            self.tb_writer.add_scalar('accuracy_diff', self.acc_history[-1] - self.base_acc_history[-1], self.episode)
        if len(self.rnd_acc_history) > 0:
            self.tb_writer.add_scalar('rnd_accuracy', self.rnd_acc_history[-1], self.episode)
            self.tb_writer.add_scalar('rand_self_indicator', self.rand_self_indicator, self.episode)

        self.tb_writer.add_scalar('improve_indicator', self.improve_indicator, self.episode)
        if policy_loss is not None:
            self.tb_writer.add_scalar('policy_loss', policy_loss, self.episode)
        if value_loss is not None:
            self.tb_writer.add_scalar('value_loss', value_loss, self.episode)
        if loss is not None:
            self.tb_writer.add_scalar('loss', loss, self.episode)
        if rewards is not None and len(rewards) > 0:
            self.tb_writer.add_scalar('reward', sum(rewards) / self.reward_limit, self.episode)
        if len(self.entropy) > 0:
            self.tb_writer.add_scalar('entropy', sum(self.entropy) / len(self.entropy), self.episode)

        self.tb_writer.add_scalar('cumulative_regret', self.regret, self.episode)
        self.episode += 1

