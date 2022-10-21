import logging
import os
import pickle
from typing import List, Union, Tuple

import torch
from fuzzywuzzy import fuzz
from torch.utils.data import SequentialSampler, DataLoader, DistributedSampler, Dataset
from transformers import PreTrainedTokenizer

from beam import Beam

from run_lm import load_and_cache_examples
from dataset import LineDataset

logger = logging.getLogger('Evaluation')


def evaluate(args, model, tokenizer, prefix="", eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, do_evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else \
        DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, drop_last=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in eval_dataloader:
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": float(perplexity)}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    return tuple(repackage_hidden(v) for v in h)


def decode_ids(idxs, tokenizer: PreTrainedTokenizer):
    codes = ""
    for idx in idxs:
        to_add = tokenizer.convert_ids_to_tokens(idx)
        if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
            if not codes.endswith(" "):
                codes += " " + to_add[1:]
            else:
                codes += to_add[1:]
        elif (
                idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                        tokenizer.pad_token_id] or
                tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
        ):
            codes += " " + to_add + " "
        else:
            codes += to_add
    return codes.strip(" ")


def load_dataset(args, tokenizer: PreTrainedTokenizer, file_type: str) -> Dataset:
    pickled_tokenized_dataset_file = f"{args.data_dir}/tokenized.pkl"
    if os.path.exists(pickled_tokenized_dataset_file):
        with open(pickled_tokenized_dataset_file, 'rb') as dataset_pickled_read:
            logger.info('Loading tokenized dataset from pickle file...')
            return pickle.load(dataset_pickled_read)

    logger.info('Tokenizing dataset (this may take a very long time!)...')
    dataset = LineDataset(tokenizer, args, file_type=file_type,
                          block_size=args.block_size - 100)
    logger.info('Saving tokenized dataset to pickle file...')
    pickle.dump(dataset, open(pickled_tokenized_dataset_file, 'wb'))
    return dataset


def eval_line_completion(args, model, tokenizer: PreTrainedTokenizer, file_type='test'):
    """
    Evaluate line level code completion on exact match and edit similarity.

    It is recommended to use single GPU because it could not be batched.
    """

    dataset = load_dataset(args, tokenizer, file_type)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    model.eval()

    if args.langs == "python":
        break_ids: List[Union[int, List[int]]] = [tokenizer.sep_token_id]
    else:
        break_ids: List[Union[int, List[int]]] = [tokenizer.convert_tokens_to_ids(t)
                                                  for t in ['Ġ;', 'Ġ}', 'Ġ{']]
    predictions: List[str] = []
    gts = []
    edit_sim = 0.0
    em = 0.0

    print("Starting evaluation...")

    for step, (inputs, gt) in enumerate(test_dataloader):

        if 0 < args.early_eval_stop <= step:
            break

        inputs = inputs.to(args.device)
        with torch.no_grad():
            beam_size = 5
            soft_max = torch.nn.LogSoftmax(dim=-1)
            outputs = model(inputs[:, :-1])[1]
            p = []
            zero: torch.LongTensor = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                if args.model_type == "rnn":
                    past_hidden = tuple(x[:, i:i + 1].expand(-1, beam_size, -1).contiguous()
                                        for x in outputs)
                else:
                    past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0)
                            if type(x) == tuple else x for x in outputs]
                    past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in past]
                beam = Beam(beam_size, inputs[i][-1].cpu().data, break_ids)

                for _ in range(100):
                    if beam.done():
                        break
                    input_ids: torch.LongTensor = beam.get_current_state()
                    if args.model_type == "rnn":
                        outputs: torch.Tensor = model(input_ids, hidden=repackage_hidden(past_hidden))
                    else:
                        outputs: torch.Tensor = model(input_ids, past_key_values=past_hidden)
                    out: torch.Tensor = soft_max(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    if args.model_type == "rnn":
                        past_hidden: Tuple[torch.Tensor] = tuple(
                            x.data.index_select(1, beam.get_current_origin()).contiguous()
                            for x in outputs[1])
                    else:
                        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0)
                                if type(x) == tuple else x for x in outputs[1]]
                        past_hidden = tuple(x.data.index_select(1, beam.get_current_origin())
                                            for x in past)
                hyp = beam.get_hypothesis(beam.get_final())
                pred = beam.build_target_tokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1)
                        for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.langs == "python":
                    text = decode_ids(t, tokenizer).strip("<EOL>").strip()
                else:
                    text = decode_ids(t, tokenizer).strip("{").strip()

                predictions.append(text)
                gts.append(gt[0])
                edit_sim += fuzz.ratio(text, gt[0])
                em += 1 if text == gt[0] else 0
        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")

    saved_file = os.path.join(args.output_dir, "predictions_line.txt")
    with open(saved_file, "w") as f:
        f.write('\n'.join(predictions) + '\n')

    logger.info(f"Test %d samples", len(predictions))
    logger.info(f"Edit sim: %f, EM: %f", edit_sim / len(predictions), em / len(predictions))
