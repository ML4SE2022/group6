all:
	@echo "Please provide a target."

PYLANG=python
PYDATASET = py150
PYDATADIR=dataset/${PYDATASET}/token_completion
PYLITFILE=dataset/${PYDATASET}/literals.json
PYOUTPUTDIR=save/${PYDATASET}
PYPRETRAINDIR=microsoft/CodeGPT-small-py-adaptedGPT2
PYLOGFILE=logs/token_completion_py150_python_train.log

# For evaluation
CHECKPOINT=${PYOUTPUTDIR}/checkpoint-last
LANG=${PYLANG}
DATASET=${PYDATASET}
DATADIR=dataset/${PYDATASET}/token_completion
LITFILE=${PYLITFILE}
OUTPUTDIR=save/eval/${DATASET}
LOGFILE=logs/token_eval_${LANG}.log

JAVAPRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2

TSLANG=typescript
TSDATASET=typescriptAxolotl
TSDATADIR=dataset/${TSDATASET}/token_completion
TSLITFILE=dataset/${TSDATASET}/literals.json
TSOUTPUTDIR=save/${TSDATASET}
TSLOGFILE=logs/token_completion_java_typescript_train.log

JSLANG=javascript
JSDATASET=javascriptAxolotl
JSDATADIR=dataset/${JSDATASET}/token_completion
JSLITFILE=dataset/${JSDATASET}/literals.json
JSOUTPUTDIR=save/${JSDATASET}
JSLOGFILE=logs/token_completion_java_javascript_train.log

PYJSOUTPUTDIR=save/small-py-adapted_${JSDATASET}
PYTSOUTPUTDIR=save/small-py-adapted_${TSDATASET}

PYJSLOGFILE=logs/token_completion_python_on_javascript.log
PYTSLOGFILE=logs/token_completion_python_on_typescript.log


# For cross training
CROSSOUTPUTDIR=save/cross/js-ts-py
CROSSPRETRAINDIR=save/cross/js-ts/checkpoint-last
CROSSLOGFILE=logs/cross_js_ts_py_train.log

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
		--logging_steps=500 \
		--save_steps=10000 \
		--seed=42 \
		--overwrite_output_dir \
		--not_pretrain

py-train-token-on-js:
	python -m torch.distributed.launch --nproc_per_node=1 code/run_lm.py \
		--data_dir=${JSDATADIR} \
		--lit_file=${JSLITFILE} \
		--langs=${JSLANG} \
		--output_dir=${PYJSOUTPUTDIR} \
		--pretrain_dir=${PYPRETRAINDIR} \
		--log_file=${PYJSLOGFILE} \
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
		--logging_steps=100 \
		--save_steps=1000 \
		--seed=42 \
		--overwrite_output_dir \
		--not_pretrain

py-train-token-on-ts:
	python -m torch.distributed.launch --nproc_per_node=1 code/run_lm.py \
		--data_dir=${TSDATADIR} \
		--lit_file=${TSLITFILE} \
		--langs=${TSLANG} \
		--output_dir=${PYTSOUTPUTDIR} \
		--pretrain_dir=${PYPRETRAINDIR} \
		--log_file=${PYTSLOGFILE} \
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
		--logging_steps=100 \
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
		--pretrain_dir=${JAVAPRETRAINDIR} \
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
		--logging_steps=100 \
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
		--pretrain_dir=${JAVAPRETRAINDIR} \
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
		--logging_steps=100 \
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

js-ts-py-train-token:
	python -m torch.distributed.launch --nproc_per_node=1 code/run_lm.py \
		--data_dir=${PYDATADIR} \
		--lit_file=${PYLITFILE} \
		--langs=${PYLANG} \
		--output_dir=${CROSSOUTPUTDIR} \
		--pretrain_dir=${CROSSPRETRAINDIR} \
		--log_file=${CROSSLOGFILE} \
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
		--logging_steps=100 \
		--save_steps=1000 \
		--seed=42 \
		--overwrite_output_dir \
		--not_pretrain


eval-all-py:
	python -u code/run_lm.py \
		--data_dir=${PYDATADIR} \
		--lit_file=${PYLITFILE} \
		--langs=${PYLANG} \
		--output_dir=save/cross/eval-all-py \
		--pretrain_dir=save/cross/js-ts-py/checkpoint-last \
		--log_file=logs/eval_all_py.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000

eval-js-ts-on-py:
	python -u code/run_lm.py \
		--data_dir=${PYDATADIR} \
		--lit_file=${PYLITFILE} \
		--langs=${PYLANG} \
		--output_dir=save/cross/eval-js-ts-on-py \
		--pretrain_dir=save/cross/js-ts/checkpoint-last \
		--log_file=logs/eval_js_ts_on_py.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000

eval-all-js:
	python -u code/run_lm.py \
		--data_dir=${JSDATADIR} \
		--lit_file=${JSLITFILE} \
		--langs=${JSLANG} \
		--output_dir=save/cross/eval-all-js \
		--pretrain_dir=save/cross/js-ts-py/checkpoint-last \
		--log_file=logs/eval_all_js.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000

eval-js-ts-on-js:
	python -u code/run_lm.py \
		--data_dir=${JSDATADIR} \
		--lit_file=${JSLITFILE} \
		--langs=${JSLANG} \
		--output_dir=save/cross/eval-js-ts-on-js \
		--pretrain_dir=save/cross/js-ts/checkpoint-last \
		--log_file=logs/eval_js_ts_on_js.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000

eval-all-ts:
	python -u code/run_lm.py \
		--data_dir=${TSDATADIR} \
		--lit_file=${TSLITFILE} \
		--langs=${TSLANG} \
		--output_dir=save/cross/eval-all-ts \
		--pretrain_dir=save/cross/js-ts-py/checkpoint-last \
		--log_file=logs/eval_all_ts.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000

eval-js-ts-on-ts:
	python -u code/run_lm.py \
		--data_dir=${TSDATADIR} \
		--lit_file=${TSLITFILE} \
		--langs=${TSLANG} \
		--output_dir=save/cross/eval-js-ts-on-ts \
		--pretrain_dir=save/cross/js-ts/checkpoint-last \
		--log_file=logs/eval_js_ts_on_ts.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000


eval-none-py:
	python -u code/run_lm.py \
		--data_dir=${PYDATADIR} \
		--lit_file=${PYLITFILE} \
		--langs=${PYLANG} \
		--output_dir=save/not-tuned/eval-none-py \
		--pretrain_dir=${PYPRETRAINDIR} \
		--log_file=logs/eval_all_py.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000

eval-none-js:
	python -u code/run_lm.py \
		--data_dir=${JSDATADIR} \
		--lit_file=${JSLITFILE} \
		--langs=${JSLANG} \
		--output_dir=save/not-tuned/eval-none-js \
		--pretrain_dir=${PYPRETRAINDIR} \
		--log_file=logs/eval_all_py.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000

eval-none-ts:
	python -u code/run_lm.py \
		--data_dir=${TSDATADIR} \
		--lit_file=${TSLITFILE} \
		--langs=${TSLANG} \
		--output_dir=save/not-tuned/eval-none-ts \
		--pretrain_dir=${PYPRETRAINDIR} \
		--log_file=logs/eval_all_py.log \
		--model_type=gpt2 \
		--block_size=1024 \
		--do_eval \
		--per_gpu_eval_batch_size=16 \
		--logging_steps=100 \
		--seed=42 \
		--early_eval_stop=1000