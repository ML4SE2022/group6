# Group 6 -- Code Completion (token level)


Here is the introduction and pipeline for token level code completion task.

## Task Definition

Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.

Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.


## Docker

## Prerequisites

- Install NVidia drivers from [here](https://www.nvidia.com/Download/index.aspx?lang=en-us)
- Install docker
- Make sure `nvidia-container-toolkit` is installed
- Any nvidia driver issues are left as an exersise to the reader
- 

#### Dataset
Build dataset creator with the following command: (this is quick)
> ```docker build -t dataset -f dataset.Dockerfile .```

Run dataset collector with the following command: (this is slow)
> `docker run --mount type=bind,source=$(pwd)/dataset,target=/dataset dataset`

#### Training

Build the training image with the following command where `CUDA_VERSION` can be one of `cu116, cu113, cu102, cpu`: (this is quite slow)
> `docker build -t token_completion . --build-arg CUDA_VERSION=[CUDA_VERSION]`

Run the trainer with the following command where `MAKE_TARGET` is a target from [Makefile](Makefile): 
> `docker run --gpus all --mount type=bind,source=$(pwd)/dataset--mount type=bind,source=$(pwd)/save,target=/save,target=/save --mount type=bind,source=$(pwd)/logs,target=/logs token_completion [MAKE_TARGET]`

Or run everything we got!

> `docker run --gpus all --mount type=bind,source=$(pwd)/dataset--mount type=bind,source=$(pwd)/save,target=/save,target=/save --mount type=bind,source=$(pwd)/logs,target=/logs --entrypoint bash token_completion [eval-all.sh | run-all.sh]`

## Local Installation

First install the dependencies using poetry:

```bash
poetry install
```

## Dataset
TODO add links

## Fine-tuned models
TODO add links


## Running the code

Most of the commands used to generate our results can be found as targets in the `Makefile`. To replicate our results:

- If you want to fine-tune the pre-trained models, first make sure you have all the datasets downloaded. Then, run the `run-all.sh` bash script.
- If you want to evaluate, do the previous step or make sure you have downloaded our fine-tuned models. Then, run the `eval-all.sh` bash script.
