""" Get the training data for the adapter """

# We expect the input data following the format:
# (id, input, output, ctxs, dataset_name)

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
import torch

import argparse
import json
from functools import partial
from utils import (
    TASK_INST_CONTEXT,
    TASK_INPUT,
    PROMPT_DICT,
    PROMPT,
    PROMPT_CONTEXT,
    SHOT_TEMPLATE,
    SHOT_TEMPLATE_CONTEXT,
)

INPUT_FORMAT = {
    "context": "Context:\nDocument 1: {ctxs}\nQuestion:\n{input}",
    "no_context": "Question:\n{input}",
}

def format_instruction_input(instruction, input, ctxs, label):
    inputs = []
    if len(ctxs) == 0:
        input_no_ctx = INPUT_FORMAT["no_context"].format(input=input)
        inputs.append(input_no_ctx)
    else:
        for ctx in ctxs:
            # we only use 1 context during training Adapter
            input_ctx = INPUT_FORMAT["context"].format(ctxs=ctx, input=input)
            inputs.append(input_ctx)
    inputs = [PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input) for input in inputs]
    inputs_label = [input + label for input in inputs]
    
    return (inputs, inputs_label)

def format_icl_input(input, ctxs, label, task, shots_num):
    task_input = TASK_INPUT[task]
    # get the shots
    shot_example = json.load(open('data_creation/eval_shots.json', 'r'))[task][:shots_num]
    shots = ""
    for shot in shot_example:
        if len(ctxs) > 0:
            shots += SHOT_TEMPLATE_CONTEXT.format(task_input=task_input, **shot)
        else:
            shots += SHOT_TEMPLATE.format(task_input=task_input, **shot)
    inputs = []
    if len(ctxs) == 0:
        input_no_ctx = PROMPT.format(shots=shots, input=input, task_input=task_input)
        inputs.append(input_no_ctx)
    else:
        for ctx in ctxs:
            # we only use 1 context during training Adapter
            ctx = "Document 1: " + ctx
            input_ctx = PROMPT_CONTEXT.format(shots=shots, context=ctx, input=input, task_input=task_input)
            inputs.append(input_ctx)
    inputs_label = [input + label for input in inputs]

    return (inputs, inputs_label)

def get_log_lm_probability(model, tokenized_prompt, tokenized_example):

    tokenized_example = {k: torch.tensor(v).to(model.device) for k, v in tokenized_example.items()}
    tokenized_prompt = {k: torch.tensor(v).to(model.device) for k, v in tokenized_prompt.items()}
    input_ids = tokenized_example['input_ids'] # (num_ctxs, seq_length)
    label_ids = input_ids[:, tokenized_prompt['input_ids'].shape[1]:] # (num_ctxs, label_length)
    attention_mask = tokenized_example['attention_mask'] # (num_ctxs * num_label, seq_length)
    
    with torch.no_grad():
        output = model(**tokenized_example)
    logits = output.logits[:, tokenized_prompt['input_ids'].shape[1] - 1: -1, :]
    probabilities = torch.softmax(logits, dim=-1)
    
    label_probs = torch.gather(probabilities, 2, label_ids.unsqueeze(-1)).squeeze(-1)

    # get the sum of the log probabilities
    log_probs = torch.sum(torch.log(label_probs), dim=-1)

    return log_probs # (num_ctxs, )

def get_log_prob_LSR_score(log_prob_context, temperature=0.01):

    probs = torch.exp(log_prob_context)
    # because the probs are too small, we need to scale it
    probs = probs / temperature

    # prob is too small, log_sum_exp_prob override the prob
    log_sum_exp_prob = torch.logsumexp(probs, dim=0)
    log_prob_lsrs = probs - log_sum_exp_prob
    
    return log_prob_lsrs # (num_ctxs,)

def generate(model, input_data, temperature=0.01):

    log_prob_context = get_log_lm_probability(model,
                                              input_data['tokenized_prompt'],
                                              input_data['tokenized_example'])
    log_prob_lsrs = get_log_prob_LSR_score(log_prob_context, temperature)
    return {'log_prob_context': log_prob_context.cpu(),
            'log_prob_lsrs': log_prob_lsrs.cpu()}

def convert_ctxs(example):
    ctxs = []
    for ctx in example['ctxs']:
        ctxs.append(f"(Title: {ctx['title']}) {ctx['text']}")

    return {'ctxs': ctxs}

def tokenize_func(input_data, tokenizer, max_length, icl=False, shots_num=5):
    label = input_data['output']
    input = input_data['input']
    ctxs = input_data['ctxs']

    if icl:
        label += "\n"  # for In-context learning, we take newline character as eos token
        inputs, inputs_label = format_icl_input(input, ctxs, label, input_data['dataset_name'], shots_num)
    else:
        instruction = TASK_INST_CONTEXT[input_data['dataset_name']]
        label += tokenizer.eos_token
        inputs, inputs_label = format_instruction_input(instruction, input, ctxs, label)

    tokenized_example = tokenizer(inputs_label,
                                  return_tensors='pt',
                                  padding=True,
                                  max_length=max_length,
                                  truncation=True)
    tokenized_prompt = tokenizer(inputs,
                                 return_tensors='pt',
                                 padding=True,
                                 max_length=max_length,
                                 truncation=True)
    
    return {'tokenized_example': tokenized_example,
            'tokenized_prompt': tokenized_prompt}

def parse_args():

    parser = argparse.ArgumentParser(description='Adapter Training Data Creation')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--input', type=str, help='input file, should be in jsonl format')
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature')
    parser.add_argument('--output', type=str, help='output file for training adapter')
    parser.add_argument('--use_flash-attn', action='store_true', help='use flash attention')
    parser.add_argument('--seed', type=int, default=42, help='set seed for reproducibility')
    parser.add_argument('--num_processes', type=int, default=20, help='number of processes to use')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='max length of the input')
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
    args = parser.parse_args()
    return args

def main(args):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="flash_attention_2" if args.use_flash_attn else "default"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    if tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
        print(f"Added {num_added_tokens} tokens to the tokenizer.")
        model.resize_token_embeddings(len(tokenizer))
    # load the input data
    print("Loading input data...")
    # input_data = load_jsonlines(args.input)
    input_data = load_dataset('json', data_files=args.input)['train']

    input_data = input_data.map(convert_ctxs,
                                num_proc=args.num_processes,
                                desc="Converting context...")

    print("Tokenizing input data...")
    encode_func = partial(tokenize_func,
                          tokenizer=tokenizer,
                          max_length=args.max_seq_length,
                          icl=args.in_context_learning,
                          shots_num=args.shots_num)
    input_data = input_data.map(encode_func,
                                num_proc=args.num_processes,
                                desc="Tokenizing...")
    
    model.eval()
    encode_func = partial(generate, model, temperature=args.temperature)
    input_data = input_data.map(encode_func,
                                remove_columns=['tokenized_example', 'tokenized_prompt'],
                                desc="Getting probability...")
    
    # save the output data
    print(f"Saving output data to {args.output}...")
    input_data.to_json(args.output, orient='records', lines=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
