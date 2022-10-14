all: 
	@echo "Please provide a target."

LANG = python
DATASET = py150
DATADIR = dataset/${DATASET}/line_completion
LITFILE = dataset/${DATASET}/literals.json
OUTPUTDIR = save/${DATASET} # The output directory where the model predictions and checkpoints will be written. 
PRETRAINDIR = microsoft/CodeGPT-small-py-adaptedGPT2
LOGFILE = logs/completion_py150_eval.log

inference-python: 
	python -u code/run_lm.py \
			--data_dir=${DATADIR} \
			--lit_file=${LITFILE} \
			--langs=${LANG} \
			--output_dir=${OUTPUTDIR} \
			--pretrain_dir=${PRETRAINDIR} \
			--log_file=${LOGFILE} \
			--model_type=gpt2 \
			--block_size=1024 \
			--eval_line \
			--logging_steps=5 \
			--seed=42 \
			--save_steps=1000 \
			--early_eval_stop=20


train-python:
	python -u code/run_lm.py \
			--data_dir=${DATADIR} \
			--lit_file=${LITFILE} \
			--langs=${LANG} \
			--output_dir=${OUTPUTDIR} \
			--pretrain_dir=${PRETRAINDIR} \
			--log_file=${LOGFILE} \
			--model_type=gpt2 \
			--block_size=1024 \
			--logging_steps=10 \
			--seed=42 \
			--save_steps=1000 \
			--early_eval_stop=30 \
			--do_train \
			--overwrite_output_dir


evaluator-example:
	python evaluator/evaluator.py \
			-a=evaluator/answers.json \
			-p=evaluator/predictions.txt