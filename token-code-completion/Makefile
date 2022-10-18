all: 
	@echo "Please provide a target."

LANG=java
DATASET=javaCorpus
DATADIR=dataset/${DATASET}/token_completion
LITFILE=dataset/${DATASET}/literals.json
OUTPUTDIR=save/${DATASET} # The output directory where the model predictions and checkpoints will be written. 
PRETRAINDIR=microsoft/CodeGPT-small-java
LOGFILE=logs/token_completion_java_train.log

train-token:
	python -m torch.distributed.launch --nproc_per_node=1 code/run_lm.py \
		--data_dir=${DATADIR} \
		--lit_file=${LITFILE} \
		--langs=${LANG} \
		--output_dir=${OUTPUTDIR} \
		--pretrain_dir=${PRETRAINDIR} \
		--log_file=${LOGFILE} \
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
		--seed=42 \

evaluation-example:
	python evaluator/evaluator.py -a=evaluator/answers.txt -p=evaluator/predictions.txt

