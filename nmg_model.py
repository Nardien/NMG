from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (BertConfig,
                        PreTrainedModel)
from transformers.modeling_bert import (
                        BertAttention, BertLayerNorm, BertPreTrainedModel,
                        BertModel, BertIntermediate, BertOutput)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GeneratingMasksAC(BertPreTrainedModel):
    def __init__(self, config):
        super(GeneratingMasksAC,self).__init__(config)
        if config.model_type == 'bert':
            self.bert = BertModel(config=config)
        else:
            self.bert = None

        # Reload config (Since it's bert.., I think there is a way to modify
        # this more simple)
        if self.bert is not None:
            config = BertConfig.from_pretrained("bert-base-uncased")
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        self.config = config
        self.transformer = BertAttention(config)

        self.policy1 = nn.Linear(config.hidden_size, 128)
        self.policy2 = nn.Linear(128, 1)

        # Value Part #
        self.value1 = nn.Linear(config.hidden_size, 128)
        self.value2 = nn.Linear(128, 1)

        #self.apply(self._init_weights)
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transformer_forward(self, _input, attention_mask, head_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        x = self.transformer(_input.detach_(),
                             attention_mask=extended_attention_mask,
                             head_mask=head_mask)
        x = x[0]
        return x

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None, visualize=False,
                args=None):
        if args.continual:
            # input_ids should be embedding
            _input = input_ids.detach()
        else:
            outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)

            _input = outputs[0].detach()
        bsz = _input.shape[0]
        x = self.transformer_forward(_input, attention_mask, head_mask)

        pi = self.policy1(x)
        if visualize: middle_features = pi.squeeze(0)
        pi = gelu(pi)
        logit = self.policy2(pi).squeeze(-1)

        v_input = _input
        x = torch.mul(v_input, attention_mask.unsqueeze(-1))
        x = torch.div(x.sum(1), attention_mask.sum(-1).unsqueeze(-1))
        
        x = self.value1(x)
        x = gelu(x)
        value = self.value2(x).squeeze(-1)
        outputs = (logit, value) + (_input,)

        if visualize: outputs = outputs + (middle_features,)
        return outputs
