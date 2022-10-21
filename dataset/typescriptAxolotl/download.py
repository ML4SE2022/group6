import argparse
import os.path
import re
from io import BytesIO
from token import STRING, NUMBER, NEWLINE, NL, COMMENT, INDENT, ENCODING, ENDMARKER
from tokenize import tokenize
from tqdm import tqdm
from datasets import load_dataset
import jsbeautifier
SUBSET_SIZE = 10000
SHUFFLE_SEED = 42
FILES_OUTPUT_DIR = 'typescript_files'
DATASET_NAME = 'typescript-small'

FILES_DATA_DIR = f'{FILES_OUTPUT_DIR}/data'
FILES_DATA_TRAIN_DIR = f'{FILES_DATA_DIR}/train'
FILES_DATA_TEST_DIR = f'{FILES_DATA_DIR}/test'
OUTPUT_FILE_TRAIN_TXT = f'{FILES_OUTPUT_DIR}/{DATASET_NAME}_train.txt'
OUTPUT_FILE_TEST_TXT = f'{FILES_OUTPUT_DIR}/{DATASET_NAME}_eval.txt'
OUTPUT_FILE_DEV_TXT = f'{FILES_OUTPUT_DIR}/{DATASET_NAME}_dev.txt'
OUTPUT_FILE_TRAIN_NO_DEV_TXT = f'{FILES_OUTPUT_DIR}/{DATASET_NAME}_train_no_dev.txt'


def download(args):
    if os.path.exists(FILES_OUTPUT_DIR):
        if not args.overwrite:
            print(f'Output files dir already exists: {FILES_OUTPUT_DIR}')
            print(f'Do you want to continue? (This will remove all the files) [y/n]')
            choice = input().lower()
            if choice.startswith('y'):
                print('Removing files...')
                os.system(f'rm -rf {FILES_OUTPUT_DIR}')
            else:
                print('Exiting...')
                return exit(0)
        else:
            print('Removing files existing download...')
            os.system(f'rm -rf {FILES_OUTPUT_DIR}')

    print("Creating directories...")
    os.makedirs(FILES_DATA_TRAIN_DIR, exist_ok=True)
    os.makedirs(FILES_DATA_TEST_DIR, exist_ok=True)

    opts = jsbeautifier.default_options()
    # opts.indent_size = 2
    # opts.space_in_empty_paren = True
    opts.brace_style = ['collapse']
    opts.break_chained_methods = False
    opts.e4x = True
    # Create the train dataset
    print("Loading training dataset")
    dataset_train = load_dataset("kevinjesse/ManyTypes4TypeScript", streaming=True, split="train")
    shuffled_train = dataset_train.shuffle(seed=SHUFFLE_SEED)
    iterator = iter(shuffled_train)

    print("Writing to {}".format(OUTPUT_FILE_TRAIN_TXT))
    with open(OUTPUT_FILE_TRAIN_TXT, 'w') as f:
        for i in tqdm(range(SUBSET_SIZE)):
            next_file = next(iterator)
            # print(next_file)
            filename = f'{i:05d}.ts'
            filepath = f'{FILES_DATA_DIR}/train/{filename}'
            with open(filepath, 'w') as f_ts:
                f_ts.write(jsbeautifier.beautify(' '.join(next_file['tokens']), opts))
                # f_ts.write(' '.join(next_file['tokens']).replace('; ', ';\n'))
            f.write(filepath + '\n')

    # Create the test dataset
    print("Loading test dataset")
    dataset_test = load_dataset("kevinjesse/ManyTypes4TypeScript", streaming=True, split="test")
    shuffled_test = dataset_test.shuffle(seed=SHUFFLE_SEED*2)
    iterator = iter(shuffled_test)

    print("Writing to {}".format(OUTPUT_FILE_TEST_TXT))
    with open(OUTPUT_FILE_TEST_TXT, 'w') as f:
        for i in tqdm(range(SUBSET_SIZE)):
            next_file = next(iterator)
            filename = f'{i:05d}.ts'
            filepath = f'{FILES_DATA_DIR}/test/{filename}'
            with open(filepath, 'w') as f_ts:
                f_ts.write(jsbeautifier.beautify(' '.join(next_file['tokens']), opts))
                # f_ts.write(' '.join(next_file['tokens']))
            f.write(filepath + '\n')


def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ['\'', '"', '`']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    # This does not exist in TS, but this should not break this regex or the code below
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    # if start_quote in str_quote_options[:2]:
    #     return ""
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    # return (
    #     f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
    #     if str_lit in lits['str']
    #     else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    # )
    # TODO: Determine how to create the literals.json file
    # For now we just always return a simple string lit tag
    return f"{qualifier}{start_quote}<STR_LIT>{end_quote}"


def ts_tokenize(args, file_name, file_type):
    with open(file_name, 'r') as file_paths_f:
        file_paths = file_paths_f.readlines()

    with open(os.path.join(args.output_dir, f"{file_type}.txt"), 'w') as wf:
        for ct, path in tqdm(enumerate(file_paths), total=len(file_paths)):
            try:
                code_file = open(path.strip(), 'r')
                code = code_file.read()
                code_file.close()
                token_gen = tokenize(BytesIO(bytes(code, 'utf8')).readline)

                out_tokens = []
                prev_eol = False
                for toknum, tokval, _, _, _ in token_gen:
                    if toknum == STRING:
                        add_token = process_string(tokval)
                        out_tokens.append(add_token)
                        prev_eol = False
                    elif toknum == NUMBER:
                        # if tokval in lits['num']:
                        #     out_tokens.append(f"<NUM_LIT:{tokval}>")
                        # else:
                        #     out_tokens.append(f"<NUM_LIT>")
                        # TODO Determine how to create the literals.json file
                        out_tokens.append(f"<NUM_LIT>")
                        prev_eol = False
                    elif toknum in [NEWLINE, NL]:
                        if not prev_eol:
                            out_tokens.append('<EOL>')
                            prev_eol = True
                    elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                        continue
                    else:
                        out_tokens.append(tokval)
                        prev_eol = False
                if out_tokens[0] == '<EOL>':
                    out_tokens = out_tokens[1:]
                if out_tokens[-1] == '<EOL>':
                    out_tokens = out_tokens[:-1]
                out_tokens = ["<s>"] + out_tokens + ['</s>']
                out = " ".join(out_tokens)
                wf.write(out + '\n')
            except Exception as e:
                pass
                # print(f"Error processing {path.strip()}: {e}")

def preprocess(args):
    print("Preprocessing...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_paths = open(OUTPUT_FILE_TRAIN_TXT).readlines()[:int(-0.05 * SUBSET_SIZE)]
    dev_paths = open(OUTPUT_FILE_TRAIN_TXT).readlines()[int(-0.05 * SUBSET_SIZE):]
    wf = open(OUTPUT_FILE_TRAIN_NO_DEV_TXT, "w")
    for path in train_paths:
        wf.write(path)
    wf.close()
    wf = open(OUTPUT_FILE_DEV_TXT, "w")
    for path in dev_paths:
        wf.write(path)
    wf.close()

    print("Tokenizing training set")
    ts_tokenize(args, file_name=OUTPUT_FILE_TRAIN_NO_DEV_TXT, file_type="train")
    print("Tokenizing test set")
    ts_tokenize(args, file_name=OUTPUT_FILE_TEST_TXT, file_type="test")
    print("Tokenizing dev set")
    ts_tokenize(args, file_name=OUTPUT_FILE_DEV_TXT, file_type="dev")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action='store_true', help="Download the dataset if true")
    parser.add_argument("--overwrite", action='store_true', help='Force overwriting during the download step')
    parser.add_argument("--preprocess", action='store_true', help="Do the preprocessing if true")
    parser.add_argument("--output_dir", default="token_completion", type=str, help="The output directory")
    args = parser.parse_args()

    if args.download:
        download(args)

    if args.preprocess:
        preprocess(args)


if __name__ == "__main__":
    main()
