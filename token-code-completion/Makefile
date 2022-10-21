all: 
	@echo "Please provide a target."

PYLANG=python
PYDATASET = py150
PYDATADIR=dataset/${PYDATASET}/token_completion
PYLITFILE=dataset/${PYDATASET}/literals.json
PYOUTPUTDIR=save/${PYDATASET} # The output directory where the model predictions and checkpoints will be written.
PYPRETRAINDIR=microsoft/CodeGPT-small-py-adaptedGPT2
PYLOGFILE=logs/completion_py150_eval.log

TSLANG=typescript
TSDATASET=typescriptAxolotl
TSDATADIR=dataset/${TSDATASET}/token_completion
TSLITFILE=dataset/${TSDATASET}/literals.json
TSOUTPUTDIR=save/${TSDATASET}
TSPRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2
TSLOGFILE=logs/token_completion_java_typescript_train.log

JSLANG=javascript
JSDATASET=javascriptAxolotl
JSDATADIR=dataset/${JSDATASET}/token_completion
JSLITFILE=dataset/${JSDATASET}/literals.json
JSOUTPUTDIR=save/${JSDATASET}
JSPRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2
JSLOGFILE=logs/token_completion_java_javascript_train.log


py-train-token:
	python -m torch.distributed.launch --nproc_per_node=1 code/run_lm.py \
		--data_dir=${PYDATADIR} \
		--lit_file=${PYLITFILE} \
		--langs=${PYLANG} \
		--output_dir=${PYOUTPUTDIR} \
		--pretrain_dir=${PYPRETRAINDIR} \
		--log_file=${PYLOGFILE} \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_train \
		--gpu_per_node=1 \
		--learning_rate=8e-5 \
		--weight_decay=0.01 \
		--evaluate_during_training \
		--per_gpu_train_batch_size=2 \
		--per_gpu_eval_batch_size=4 \
		--gradient_accumulation_steps=4 \
		--num_train_epochs=5 \
		--logging_steps=1 \
		--save_steps=1000 \
		--seed=42 \
		--overwrite_output_dir \
		--not_pretrain

ts-train-token:
	python -m torch.distributed.launch --nproc_per_node=1 code/run_lm.py \
		--data_dir=${TSDATADIR} \
		--lit_file=${TSLITFILE} \
		--langs=${TSLANG} \
		--output_dir=${TSOUTPUTDIR} \
		--pretrain_dir=${TSPRETRAINDIR} \
		--log_file=${TSLOGFILE} \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_train \
		--gpu_per_node=1 \
		--learning_rate=8e-5 \
		--weight_decay=0.01 \
		--evaluate_during_training \
		--per_gpu_train_batch_size=2 \
		--per_gpu_eval_batch_size=4 \
		--gradient_accumulation_steps=4 \
		--num_train_epochs=5 \
		--logging_steps=1 \
		--save_steps=1000 \
		--seed=42 \
		--overwrite_output_dir \
		--not_pretrain

js-train-token:
	python -m torch.distributed.launch --nproc_per_node=1 code/run_lm.py \
		--data_dir=${JSDATADIR} \
		--lit_file=${JSLITFILE} \
		--langs=${JSLANG} \
		--output_dir=${JSOUTPUTDIR} \
		--pretrain_dir=${JSPRETRAINDIR} \
		--log_file=${JSLOGFILE} \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_train \
		--gpu_per_node=1 \
		--learning_rate=8e-5 \
		--weight_decay=0.01 \
		--evaluate_during_training \
		--per_gpu_train_batch_size=2 \
		--per_gpu_eval_batch_size=4 \
		--gradient_accumulation_steps=4 \
		--num_train_epochs=5 \
		--logging_steps=1 \
		--save_steps=1000 \
		--seed=42 \
		--overwrite_output_dir \
		--not_pretrain

eval-token:
	python -u code/run_lm.py \
		--data_dir=${DATADIR} \
		--lit_file=${LITFILE} \
		--langs=${LANG} \
		--output_dir=${OUTPUTDIR} \
		--pretrain_dir=${CHECKPOINT} \
		--log_file=${LOGFILE} \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42

evaluation-example:
	python evaluator/evaluator.py -a=evaluator/answers.txt -p=evaluator/predictions.txt

