import math
import os
import sys
from types import SimpleNamespace
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from schema import Schema, And, Use


def parse_args(arg_dict: Dict) -> SimpleNamespace:
    schema = Schema(
        {
            '--local_rank': Use(int),
            '--seed': Use(int),
            '--batch-size': Use(int),
            '--embed-size': Use(int),
            '--hidden-size': Use(int),
            '--clip-grad': Use(float),
            '--label-smoothing': Use(float),
            '--log-every': Use(int),
            '--max-epoch': Use(int),
            '--patience': Use(int),
            '--max-num-trial': Use(int),
            '--lr-decay': Use(float),
            '--beam-size': Use(int),
            '--sample-size': Use(int),
            '--lr': Use(float),
            '--uniform-init': Use(float),
            '--valid-niter': Use(int),
            '--dropout': Use(float),
            '--max-decoding-time-step': Use(int)
        },
        ignore_extra_keys=True
    )

    valid_arg_dict = schema.validate(arg_dict)
    arg_dict.update(valid_arg_dict)

    args = SimpleNamespace()

    def clean_name(name):
        if name.startswith('--'):
            name = name[2:]
        name = name.replace('-', '_').replace('<', '').replace('>', '').strip()

        return name

    for key, val in arg_dict.items():
        key_name = clean_name(key)
        setattr(args, key_name, val)

    return args


def init_distributed_mode(args):
    """
    Initialize distributed training parameters
        - local_rank
        - world_size

    This code snippet is adapted from fairseq.
    Copyright (c) Facebook, Inc. and its affiliates.
    """

    # if the process is launched using `torch.distributed.launch`
    if args.local_rank != -1:
        # read environment variables
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.n_gpu_per_node = int(os.environ['NGPU'])
    else:
        assert args.local_rank == -1
        args.local_rank = 0
        args.world_size = 1
        args.n_gpu_per_node = 1

    # define whether this is the master process / if we are in distributed mode
    args.is_master = args.local_rank == 0
    args.multi_gpu = args.world_size > 1

    # set GPU device
    if args.cuda:
        torch.cuda.set_device(args.local_rank)

    # initialize multi-GPU
    if args.multi_gpu:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing

    Code adapted from OpenNMT-py
    """
    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)

        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss
