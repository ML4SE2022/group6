# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Code completion (both token level and line level) pipeline in CodeXGLUE
"""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import json
from typing import Tuple, Any, List

import numpy as np
import torch
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          PreTrainedTokenizer)

from arguments import parser
from evaluation import eval_line_completion

from train import train
from dataset import TextDataset, FinetuneDataset
from model import RNNModel

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'rnn': (GPT2Config, RNNModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


def load_and_cache_examples(args, tokenizer: PreTrainedTokenizer, do_evaluate=False):
    """
    Create dataset instance for the arguments given
    Args:
        args: arguments as given as input to the program
        tokenizer: the tokenizer to use with the dataset
        do_evaluate: whether to use the FinetuneDataset or the TextDataset

    Returns:
        Dataset instance of either FinetuneDataset or TextDataset
    """
    file_type = 'dev' if do_evaluate else 'train'
    clazz = FinetuneDataset if args.not_pretrain else TextDataset
    return clazz(tokenizer, args, file_type, block_size=args.block_size)


def get_special_tokens(path: str) -> List[str]:
    """
    Get a list of special tokens in the datasets
    Args:
        path: path to the dataset with the literals. (Must be a JSON).

    Returns:
        List of strings indicating which special tokens are being used.
    """
    literals = json.load(open(path))
    return (
        ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"] +
        ["<STR_LIT:{lit}>" for literals in literals["str"]] +
        ["<NUM_LIT:{lit}>" for literals in literals["num"]] +
        ["<CHAR_LIT:{lit}>" for literals in literals["char"]]
    )


def set_seed(args):
    """
    Set the seed of the random number generation.
    This makes the research more reproducible.
    Args:
        args: the arguments as given as input to the program
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup_debugging(args):
    """
    Set up for remote debugging.
    Args:
        args: the arguments the program was started with.
    """
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()


def setup_cuda(args):
    """
    Set up torch to use GPU using CUDA if possible, else CPU
    Args:
        args: the arguments the program was started with.
    """
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device


def setup_logging(args):
    """
    Set up the logging infrastructure for the data. Logs to the console and to files
    Args:
        args: the arguments the program was started with.
    """
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.addHandler(logging.FileHandler(args.log_file))
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, "
        "world size: %s", args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1),
        args.fp16, torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    logger.info("local_rank: %d, node_index: %d, gpu_per_node: %d", args.local_rank,
                args.node_index, args.gpu_per_node)


def load_cached_data(args):
    """
    Load any checkpoint that was saved in the past, if the arguments indicate to.
    Args:
        args: the arguments the program was started with.
    """
    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("Reload model from %s, resume from %d steps", checkpoint_last, args.start_step)


def load_model_and_tokenizer(args) -> Tuple[PreTrainedTokenizer, Any]:
    """
    Load the tokenizer and the model from a file, and set it up using token literals.
    Args:
        args: the arguments from the command line to know where to load from.

    Returns:
        The tokenizer and the model that should be used to train on the data.
    """
    special_tokens = get_special_tokens(args.lit_file)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer_path = args.pretrain_dir if args.pretrain_dir else args.tokenizer_dir
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case,
                                                sep_token='<EOL>', bos_token='<s>',
                                                eos_token='</s>', pad_token='<pad>',
                                                unk_token='<|UNKNOWN|>',
                                                additional_special_tokens=special_tokens)
    if args.model_type == "rnn":
        model = model_class(len(tokenizer), 768, 768, 1)

    if args.pretrain_dir:
        # Load the pretrained models from disk
        if args.model_type == "rnn":
            model_last = os.path.join(args.pretrain_dir, 'model.pt')
            if os.path.exists(model_last):
                logger.warning(f"Loading model from {model_last}")
                model.load_state_dict(torch.load(model_last, map_location="cpu"))
        else:
            model = model_class.from_pretrained(args.pretrain_dir)
            model.resize_token_embeddings(len(tokenizer))
    else:
        args.vocab_size = len(tokenizer)
        if args.model_type != "rnn":
            config = config_class.from_pretrained(args.config_dir)
            model = model_class(config)
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def main():
    """
    Main functionality. Sets up debugging, cuda, logging, and a seed.
    It loads the model and tokenizer and trains or evaluates
    """
    args = parser.parse_args()

    setup_debugging(args)
    setup_cuda(args)
    setup_logging(args)
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    load_cached_data(args)
    model, tokenizer = load_model_and_tokenizer(args)

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of %d trainable parameters", num_params)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    # Only works on single GPU
    if args.eval_line:
        eval_line_completion(args, model, tokenizer, file_type="test")


if __name__ == "__main__":
    main()
