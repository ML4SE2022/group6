# Group 6 -- Code Completion (token level)


Here is the introduction and pipeline for token level code completion task.

## Task Definition

Predict next code token given context of previous tokens. Models are evaluated by token level accuracy.

Code completion is a one of the most widely used features in software development through IDEs. An effective code completion tool could improve software developers' productivity. We provide code completion evaluation tasks in two granularities -- token level and line level. Here we introduce token level code completion. Token level task is analogous to language modeling. Models should have be able to predict the next token in arbitary types.

## Installation

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
