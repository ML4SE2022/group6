import argparse
import os

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data path.")
parser.add_argument("--langs", default=None, type=str, required=True,
                    help="Languages to train, if all, train all languages in data_dir")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

# Other parameters
parser.add_argument("--model_type", default="gpt2", type=str,
                    help="The model architecture to be fine-tuned.")
parser.add_argument("--pretrain_dir", default="", type=str,
                    help="The pretrain directory where the pretrained model resides")
parser.add_argument("--config_dir", type=str,
                    help="config name. Required when training from scratch")
parser.add_argument("--tokenizer_dir", type=str,
                    help="Pre-trained tokenizer dir. Required when training from scratch")
parser.add_argument("--lit_file", type=str,
                    help="literals json file")
parser.add_argument("--load_name", type=str, default="pretrained",
                    help="Load pretrained model name")

parser.add_argument("--mlm", action='store_true',
                    help="Train with masked-language modeling loss instead of language modeling.")
parser.add_argument("--mlm_probability", type=float, default=0.15,
                    help="Ratio of tokens to mask for masked language modeling loss")

parser.add_argument("--cache_dir", default="", type=str,
                    help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
parser.add_argument("--block_size", default=1024, type=int,
                    help="Optional input sequence length after tokenization."
                         "The training dataset will be truncated in block of this size for training."
                         "Default to the model max input length for single sentence inputs (take into account special tokens).")
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--eval_line", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Run evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
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
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument('--logging_steps', type=int, default=1000,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=5000,
                    help="Save checkpoint every X updates steps.")
parser.add_argument('--save_total_limit', type=int, default=None,
                    help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
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
parser.add_argument('--not_pretrain', action='store_true',
                    help="use different dataset")

parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--node_index", type=int, default=-1,
                    help="node index if multi-node running")
parser.add_argument("--gpu_per_node", type=int, default=-1,
                    help="num of gpus per node")
parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

parser.add_argument('--log_file', type=str, default='')
parser.add_argument('--tensorboard_dir', type=str)

parser.add_argument('--early_eval_stop', type=int, default=-1)


def check_args(args):
    """
    Check the arguments for invalid combination of arguments.
    The program should error if any invalid combination of arguments is given.
    Args:
        args: the arguments that were given as input to this program.
    """
    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. "
                         "They must be run using the --mlm flag (masked language modeling).")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. "
                         "Use --overwrite_output_dir to overcome.")
