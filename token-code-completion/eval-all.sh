eval_model() {
  # Function to evaluate the a model (first argument) on a language (second argument), and a dataset (third argument)
  local MODEL=$1
  local LANG=$2
  local DATASET=$3

  local DATA_DIR="dataset/${DATASET}/token_completion"
  local LITERALS_FILE="dataset/${DATASET}/literals.json"
  local OUTPUT_DIR="save/dataset_${DATASET}_model_${MODEL}"
  local LOGFILE="logs/token_completion_eval_dataset_${DATASET}_model_${MODEL}.log"
  local PRETRAIN_DIR="save/${MODEL}/checkpoint-last"

  echo "Evaluating model ${MODEL} on ${LANG} dataset: ${DATASET}"

  python -u code/run_lm.py \
    --data_dir="${DATA_DIR}" \
    --lit_file="${LITERALS_FILE}" \
    --langs="${LANG}" \
    --output_dir="${OUTPUT_DIR}" \
    --pretrain_dir="${PRETRAIN_DIR}" \
    --log_file="${LOGFILE}" \
    --model_type=gpt2 \
    --block_size=1024 \
    --do_eval \
    --per_gpu_eval_batch_size=16 \
    --logging_steps=100 \
    --seed=42
}

# Pretrained on Python, finetuned on Python
(eval_model py150 python py150 || echo "Failed to evaluate model pretrained on Python and finetuned on Python on Python dataset"); \
(eval_model py150 javascript javascriptAxolotl || echo "Failed to evaluate model pretrained on Python and finetuned on Python on Javascript dataset"); \
(eval_model py150 typescript typescriptAxolotl || echo "Failed to evaluate model pretrained on Python and finetuned on Python on Typescript dataset"); \
(eval_model small-py-adapted_javascriptAxolotl python py150 || echo "Failed to evaluate model pretrained on Python and finetuned on Javascript on Python dataset"); \
(eval_model small-py-adapted_javascriptAxolotl javascript javascriptAxolotl || echo "Failed to evaluate model pretrained on Python and finetuned on Javascript on Javascript dataset"); \
(eval_model small-py-adapted_javascriptAxolotl typescript typescriptAxolotl  || echo "Failed to evaluate model pretrained on Python and finetuned on Javascript on Typescript dataset"); \
(eval_model small-py-adapted_typescriptAxolotl python py150 || echo "Failed to evaluate model pretrained on Python and finetuned on Typescript on Python dataset"); \
(eval_model small-py-adapted_typescriptAxolotl javascript javascriptAxolotl || echo "Failed to evaluate model pretrained on Python and finetuned on Typescript on Javascript dataset"); \
(eval_model small-py-adapted_typescriptAxolotl typescript typescriptAxolotl ||  echo "Failed to evaluate model pretrained on Python and finetuned on Typescript on Typescript dataset"); \
(eval_model typescriptAxolotl python py150 || echo "Failed to evaluate model pretrained on Java and finetuned on Typescript on Python dataset"); \
(eval_model typescriptAxolotl javascript javascriptAxolotl || echo "Failed to evaluate model pretrained on Java and finetuned on Typescript on Javascript dataset"); \
(eval_model typescriptAxolotl typescript typescriptAxolotl || echo "Failed to evaluate model pretrained on Java and finetuned on Javascript on Typescript dataset"); \
(eval_model javascriptAxolotl python py150 || echo "Failed to evaluate model pretrained on Java and finetuned on Javascript on Python dataset"); \
(eval_model javascriptAxolotl javascript javascriptAxolotl || echo "Failed to evaluate model pretrained on Java and finetuned on Javascript on Javascript dataset"); \
(eval_model javascriptAxolotl typescript typescriptAxolotl || echo "Failed to evaluate model pretrained on Java and finetuned on Javascript on Typescript dataset"); \
echo "Done!"




