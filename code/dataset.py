# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import logging
import os
import pickle
import gc
import json
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger('Dataset')


def get_rank_world_size(args):
    if args.local_rank == -1:
        return 0, 1
    return args.local_rank, torch.distributed.get_world_size()


def get_inputs(args, file_type: str, filename: str):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    cached_file = os.path.join(args.output_dir, filename)
    if os.path.exists(cached_file) and not args.overwrite_cache:
        if file_type == 'train':
            logger.warning("Loading features from cached file %s", cached_file)
        with open(cached_file, 'rb') as handle:
            return cached_file, pickle.load(handle)
    return cached_file, None


def get_input_ids(args, tokenizer: PreTrainedTokenizer, file_type: str) -> List[int]:
    datafile = os.path.join(args.data_dir, f"{file_type}.txt")
    if file_type == 'train':
        logger.warning("Creating features from dataset file at %s", datafile)
    with open(datafile) as f:
        data = f.readlines()

    length = len(data)
    logger.info("Data size: %d", length)
    input_ids = []
    for idx, x in enumerate(data):
        x = x.strip()
        if x.startswith("<s>") and x.endswith("</s>"):
            pass
        else:
            x = "<s> " + x + " </s>"
        try:
            input_ids.extend(tokenizer.encode(x))
        except Exception:
            pass
        if idx % (length // 10) == 0:
            percent = idx / (length // 10) * 10
            logger.warning("Load %d", percent)
    del data
    gc.collect()

    return input_ids


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_type='train', block_size=1024):
        local_rank, world_size = get_rank_world_size(args)

        cached_file, self.inputs = get_inputs(
            args, file_type, f"{file_type}_langs_{args.langs}_blocksize_{block_size}_worldsize_"
                             f"{world_size}_rank_{local_rank}")

        if self.inputs is None:
            self.inputs = []
            input_ids = self.get_input_ids(args, file_type, world_size, local_rank, tokenizer)

            length = len(input_ids)
            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i : i + block_size])
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples", local_rank, length,
                               len(self.inputs))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_input_ids(self, args, file_type: str, world_size: int, local_rank: int,
                      tokenizer: PreTrainedTokenizer) -> List[int]:
        langs = [args.langs] if args.langs == 'all' else os.listdir(args.data_dir)
        data = []
        for lang in langs:
            datafile = os.path.join(args.data_dir, lang, file_type + '.pkl')
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
                dataset = pickle.load(open(datafile, 'rb'))
                data.extend(['<s> ' + ' '.join(x['function'].split()) + ' </s>'
                             for idx, x in enumerate(dataset) if idx % world_size == local_rank])

        length = len(data)
        logger.warning("Data size: %d", length)
        input_ids = []
        for idx, x in enumerate(data):
            try:
                input_ids.extend(tokenizer.encode(x))
            except Exception:
                pass
            if idx % (length // 10) == 0:
                percent = idx / (length // 10) * 10
                logger.warning("Rank %d, load %d", local_rank, percent)
        del data
        gc.collect()

        return input_ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class FinetuneDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_type='train', block_size=1024):
        local_rank, world_size = get_rank_world_size(args)
        cached_file, self.inputs = get_inputs(
            args, file_type, f"{file_type}_blocksize_{block_size}_wordsize_{world_size}"
                             f"_rank_{local_rank}")

        if self.inputs is None:
            self.inputs = []
            input_ids = get_input_ids(args, tokenizer, file_type)

            length = len(input_ids) // world_size
            logger.info(f"Tokens: ", length * world_size)
            input_ids = input_ids[local_rank * length: (local_rank + 1) * length]

            for i in range(0, length-block_size, block_size):
                self.inputs.append(input_ids[i: i + block_size])
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples", local_rank, length,
                               len(self.inputs))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class EvalDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []
            input_ids = get_input_ids(args, tokenizer, file_type)
            logger.info(f"Tokens: %d", len(input_ids))
            self.split(input_ids, tokenizer, logger, block_size=block_size)
            del input_ids
            gc.collect()

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def split(self, input_ids: List[int], tokenizer: PreTrainedTokenizer, block_size=1024):
        i = 0
        while i < len(input_ids):
            sample = input_ids[i: i+block_size]
            if len(sample) == block_size:
                for j in range(block_size):
                    if tokenizer.convert_ids_to_tokens(sample[block_size-1-j])[0] == '\u0120':
                        break
                    if sample[block_size-1-j] in [tokenizer.bos_token_id, tokenizer.eos_token_id,
                                                  tokenizer.sep_token_id]:
                        if sample[block_size-1-j] != tokenizer.bos_token_id:
                            j -= 1
                        break
                if j == block_size-1:
                    print(tokenizer.decode(sample))
                    exit()
                sample = sample[: block_size-1-j]

            i += len(sample)
            pad_len = block_size-len(sample)
            sample += [tokenizer.pad_token_id]*pad_len
            self.inputs.append(sample)

            if len(self.inputs) % 10000 == 0:
                logger.info(f"%d samples", len(self.inputs))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class LineDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_type='test', block_size=924):
        datafile = os.path.join(args.data_dir, f"{file_type}.json")
        with open(datafile) as f:
            datas = f.readlines()

        length = len(datas)
        logger.info("Data size: %d", length)
        self.inputs = []
        self.gts = []
        for data in datas:
            data = json.loads(data.strip())
            self.inputs.append(tokenizer.encode(data["input"])[-block_size:])
            self.gts.append(data["gt"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]
