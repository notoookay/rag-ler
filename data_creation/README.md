## Data Process

### 1. Training data

We use instruction fine-tuning data and QA data to train our LM, instruction fine-tuning data items are sampled from [tulu_v2](https://github.com/allenai/open-instruct/tree/main), you can use python scripts in the [train](./process_data/train) to pre-process the data items for consistent format, then combine the data items have processed, finally use [llm_train.py](./llm_train.py) to construct the LM instruction-tuning data.

#### LM training data should follow the format below:

```
{
    'id' (str): item id,
    'instruction' (str): task_instruction,
    'input' (str): question, choices (for some tasks),
    'ctxs' (list): a list of contexts,
    'output' (str): answer,  # choose the first answer
    'dataset_name' (str): dataset name
}
```

For **NewsQA**, you need to first follow the instruction in [newsqa repo](https://github.com/Maluuba/newsqa) to construct the initial dataset.

You can also download our LM training data from [TODO: fill this link](no_links)

#### Re-ranker training data should follow the format below:

```
{
    'id' (str): item id,
    'input' (str): user input,
    'ctxs' (list): a list of contexts,
    'output' (str): answer,
    'dataset_name' (str): dataset name
}
```

It's almost same to LM training data format except `instruction` field. The `ctxs` field is retrieved by retriever, to get the complete training data of re-ranker, you need to retrieve the `ctxs` with `passage_retrieval.sh` in [scripts](../scripts) after processing.

After aligning the data format and combining data items , you need to use [reranker_train.py](./reranker_train.py) to get the **LSR** score, which may need 60 GB to 110 GB memory, so take care of your GPU usage.

You can also download our Re-ranker training data from [TODO: fill this link](no_links)

**Note: ctxs format in training data should already be converted to: `(Title: {title}) {text}`, but in evaluation data, it should follow the format with retriever.**

#### LM training instruction

During training, we use the instruction format below:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

### 2. Inference data

We use evaluation data from [KILT](https://github.com/facebookresearch/KILT/tree/main), in which data formats are consistent. You can use the python scripts in [eval](./process_data/eval) to pre-process the inference/evaluation data.

**Note: please pre-process the eval data before retrieval and evaluation.**

During inference, the data format should follow the format below:

```
{
    'id' (str): item id,
    'input' (str): user input,
    'ctxs' (list): a list of contexts,  # if you want to provide context
    'outputs' (str): answer,  # if you need to evaluate
    'dataset_name' (str): dataset name
}
```

#### Test split evaluation

For evaluation on test split, you need to convert the result file to KILT challenge format in the [eval.ai](https://eval.ai/web/challenges/challenge-page/689/submission), use [this script](./process_data/test/kilt.py) in the `process_data/test`. After that, you can submit the converted file.
