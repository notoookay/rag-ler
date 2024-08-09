""" Inference LLMs with or without context """

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
    LlamaTokenizer,
    LlamaTokenizerFast,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
)
import torch
from datasets import load_dataset
import numpy as np
from logging import getLogger
import mlflow

from functools import partial
import argparse
from metrics import (exact_match_score,
                     qa_f1_score,
                     metric_max_over_ground_truths,)
from datetime import datetime
import logging
import json

from utils import (
    TASK_INST,
    TASK_INST_CONTEXT,
    TASK_INPUT,
    PROMPT_DICT,
    PROMPT,
    PROMPT_CONTEXT,
    SHOT_TEMPLATE,
    SHOT_TEMPLATE_CONTEXT,
)

# TODO: delete this
LLAMA2_CHAT_PROMPT_DICT = {
    "prompt_no_context": "[INST] {instruction}\n\nQuestion:\n{question} [/INST]",
    "prompt_context": "[INST] {instruction}\n\nContext:\n{context}\n\nQuestion:\n{question} [/INST]",
}

def convert_ctxs(example):
    ctxs = []
    for ctx in example['ctxs']:
        ctxs.append(f"(Title: {ctx['title']}) {ctx['text']}")

    return {"transformed_ctxs": ctxs}

def create_instruction_prompt(example, chat_prompt, use_context=False, ctxs_num=0):

    # default instruction
    task = example['dataset_name']
    instruction = TASK_INST[task]
    task_input = TASK_INPUT[task]

    if use_context:
        if ctxs_num > len(example['ctxs']):
            raise ValueError(f"ctxs_num {ctxs_num} is larger than the number of contexts {len(example['ctxs'])}")
        
        ctxs = "Context:\n" + "\n".join([f"Document {i + 1}: {ctx}" for i, ctx in enumerate(example['transformed_ctxs'][:ctxs_num])])
        instruction = TASK_INST_CONTEXT[task]

    # TODO: delete chat_prompt
    if chat_prompt:
        prompt = (
            LLAMA2_CHAT_PROMPT_DICT["prompt_context"]
            if use_context
            else LLAMA2_CHAT_PROMPT_DICT["prompt_no_context"]
        )
        if use_context:
            prompts = prompt.format(instruction=instruction, context=ctxs, question=example["input"]) # str
        else:
            prompts = prompt.format(instruction=instruction, question=example["input"])
    else:
        input = example["input"]
        prompt = PROMPT_DICT["prompt_input"]

        if use_context:
            input = f"{ctxs}\n{task_input}:\n{input}"
        else:
            input = f"{task_input}:\n{input}"
        prompts = prompt.format(instruction=instruction, input=input) # str

    return {"prompt": prompts, "instruction": instruction}

def create_icl_prompt(example, use_context=False, ctxs_num=0, shots_num=5):
    """
    Args:
        example: the data item
        use_context: whether to use context
        shot_example: follow the same format with example, these may be sampled from original data.

    Returns:
        the list of prompts for in-context learning
    """

    task = example["dataset_name"]
    task_input = TASK_INPUT[task]
    # do NOT use batch!
    prompt = ""

    # get the shots
    shot_example = json.load(open('data_creation/eval_shots.json', 'r'))[task][:shots_num]
    shots = ""
    for shot in shot_example:
        if use_context:
            shots += SHOT_TEMPLATE_CONTEXT.format(task_input=task_input, **shot)
        else:
            shots += SHOT_TEMPLATE.format(task_input=task_input, **shot)
    
    # generate the prompt
    if use_context:
        if ctxs_num > len(example['ctxs']):
            raise ValueError(f"ctxs_num {ctxs_num} is larger than the number of contexts {len(example['ctxs'])}")
        
        ctxs = "\n".join([f"Document {i + 1}: {ctx}" for i, ctx in enumerate(example['transformed_ctxs'][:ctxs_num])])
        prompt = PROMPT_CONTEXT.format(task_input=task_input,
                                       shots=shots,
                                       input=example["input"],
                                       context=ctxs)
    else:
        prompt = PROMPT.format(task_input=task_input,
                               shots=shots,
                               input=example["input"])

    return {"prompt": prompt}

def rerank_contexts(example, model, tokenizer):
    inputs = [example['input']] * len(example['transformed_ctxs'])
    inputs = tokenizer(inputs, example['transformed_ctxs'], padding=True, truncation=True, return_tensors='pt').to(model.device)
    
    model.eval()
    with torch.no_grad():
        scores = model(**inputs).logits
    
    scores = scores.squeeze()
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_ctxs = [example['ctxs'][i] for i in indices]
    sorted_transformed_ctxs = [example['transformed_ctxs'][i] for i in indices]
    
    return {"ctxs": sorted_ctxs, "transformed_ctxs": sorted_transformed_ctxs}

def generate(example, model, tokenizer, max_new_tokens, in_context_learning=False):

    # do NOT use batch, as it will add `<unk>` token after generation for padding.
    prompts = example["prompt"]
    task = example["dataset_name"]
    inputs = tokenizer(prompts,
                       return_tensors="pt",
                       padding=True,
                       truncation=True).to(model.device)
    input_ids = inputs["input_ids"]

    # Use greedy decoding
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    new_outputs = outputs[:, input_ids.shape[1]:] # only work for no batch
    preds = tokenizer.batch_decode(new_outputs)

    if in_context_learning:
        for i in range(len(preds)):
            preds[i] = preds[i].split("\n\n")[0]

    pred = preds[0] # no batch, so we only need the first one
    # manually remove the eos token
    if pred[-4:] == tokenizer.eos_token:
        pred = pred[:-4]
    
    # For fever, we find model generate 'True' or 'False', so we need to convert
    # it to lowercase
    if task == "fever":
        pred = pred.lower()
        # we find our model tends to generate 'yes' or 'no' instead of 'true' or 'false'
        if "yes" in pred:
            pred = pred.replace("yes", "true")
        else:
            pred = pred.replace("no", "false")

    return {"prediction": pred}

def evaluate(metrics_name, data):
    metrics = {k: [] for k in metrics_name}
    pred = data["prediction"]
    
    if isinstance(data["output"], str):
        labels = [data["output"]]
    else:
        labels = data["output"]

    if 'em' in metrics:
        metrics['em'].append(metric_max_over_ground_truths(exact_match_score,
                                                            pred,
                                                            labels))
    if 'accuracy' in metrics:
        metrics['accuracy'].append(max([1.0 if label in pred else 0.0 for label in labels]))

    if 'f1' in metrics:
        metrics['f1'].append(metric_max_over_ground_truths(qa_f1_score, pred, labels))

    return {k: metrics[k] for k in metrics.keys()}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate answer for the input data")
    parser.add_argument(
        "--model",
        type=str)
    parser.add_argument(
        "--tokenizer",
        type=str)
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Whether to use the slow tokenizer"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="input data only support json format now"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="suggested output file name: {model}_{task}_predictions.jsonl"
        )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes to use"
    )
    parser.add_argument(
        "--use_flash-attn",
        action="store_true",
        help="Whether to use flash attention for speed up inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Max number of new tokens to generate",
    )
    parser.add_argument(
        "--metrics",
        choices=["em", "accuracy", "f1"],
        nargs="+",
        default=["accuracy"],
        help="choose accuracy as default cause it is the most common metric for most tasks"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for generating answers"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="open-instruct",
        help="Specify the experiment name for tracking experiment with tools like"
             "`wandb`, `mlflow`, `tensorboard`."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Specify the run name for tracking experiment with tools like"
             "`wandb`, `mlflow`, `tensorboard`."
    )
    parser.add_argument(
        "--in_context_learning",
        action="store_true",
        help="Whether to use in-context learning, default is using instruction"
    )
    parser.add_argument(
        "--shots_num",
        type=int,
        default=5,
        help="The number of shots for in-context learning"
    )
    parser.add_argument(
        "--use_context",
        action="store_true",
        help="Whether to use context in the prompt"
    )
    parser.add_argument(
        "--ctxs_num",
        type=int,
        default=0,
        help="Number of contexts to use for in-context learning"
    )
    parser.add_argument(
        "--use_chat_prompt",
        action="store_true",
        help="Whether to use chat model"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Whether to rerank the given contexts",
    )
    parser.add_argument(
        "--rerank_model",
        type=str,
        help="set True when rerank is True"
    )
    parser.add_argument(
        "--rerank_tokenizer",
        type=str,
        help="set True when rerank is True"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to evaluate the generated answers"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(42)

    logger = getLogger(__name__)

    # we use mlflow to log the experiment
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(run_name=args.run_name,
                     log_system_metrics=True
                     )
    mlflow.log_params(vars(args))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2" if args.use_flash_attn else "default",
    )
    if not args.tokenizer:
        args.tokenizer = args.model

    logger.info(f"output dir: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        padding_side="left",
        use_fast=not args.use_slow_tokenizer,
    )

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        print(f"num_added_tokens: {num_added_tokens}")
        print(f"tokenizer special tokens: {tokenizer.special_tokens_map}")
        print(f"tokenizer total tokens: {len(tokenizer)}")
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    if len(tokenizer) > embeddings.weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    
    # Mistral-7B model do not add pad token with the tokenizer, add it manually
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    data = load_dataset("json", data_files=args.data, split="train")

    if args.use_context:
        data = data.map(
            convert_ctxs,
            desc="Converting contexts",
            num_proc=args.num_proc,
        )
        if args.rerank:
            rerank_model = AutoModelForSequenceClassification.from_pretrained(args.rerank_model).to('cuda')
            if not args.rerank_tokenizer:
                args.rerank_tokenizer = args.rerank_model
            rerank_tokenizer = AutoTokenizer.from_pretrained(args.rerank_tokenizer)
            data = data.map(
                partial(rerank_contexts, model=rerank_model, tokenizer=rerank_tokenizer),
                desc="Reranking contexts",
            )

    if args.in_context_learning:
        data = data.map(
            create_icl_prompt,
            desc="Creating in-context learning prompts",
            fn_kwargs={"use_context": args.use_context,
                       "ctxs_num": args.ctxs_num,
                       "shots_num": args.shots_num,},
        )
    else:  # use instruction prompt
        data = data.map(
            create_instruction_prompt,
            desc="Creating prompts",
            fn_kwargs={"chat_prompt": args.use_chat_prompt,
                       "use_context": args.use_context,
                       "ctxs_num": args.ctxs_num,},
        )

    model.eval()
    encode_func = partial(
        generate,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        in_context_learning=args.in_context_learning,
    )

    data = data.map(
        encode_func,
        desc="Generating answers",
        batched=False, # close batch to avoid `<unk>` token
    )

    # evaluate
    if args.evaluate:
        data = data.map(
            partial(evaluate, args.metrics),
            desc="Evaluating",
            num_proc=args.num_proc,
        )

        metrics = {k: np.mean(data[k]) for k in args.metrics}

        logger.info(f"Saving metrics to mlflow...")
        mlflow.log_metrics(metrics)

        logger.info(f"Metrics: {metrics}")

    logger.info(f"Saving to {args.output}...")
    # remove `context` as it's too long
    if 'transformed_ctxs' in data.column_names:
        data.remove_columns('transformed_ctxs')

    data.to_json(
        args.output,
        orient="records",
        lines=True,
        num_proc=args.num_proc,
    )

    # log the output file
    mlflow.log_artifact(args.output)

    mlflow.end_run()


if __name__ == "__main__":
    main()
