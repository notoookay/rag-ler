""" Preprocess the chosen training data. """

import argparse

from utils import load_jsonlines, save_file_jsonl, TASK_INST

def process_instruction(input_data):
    dataset_name = input_data['dataset_name']
    if dataset_name not in TASK_INST:
        return input_data
    task_instruction = TASK_INST[dataset_name]
    input_data['instruction'] = task_instruction
    return input_data
 
def process_input(input_data):
    if ('ctxs' not in input_data) or (len(input_data['ctxs']) == 0):
        input_data.pop('ctxs') # TODO: remove this later
        return input_data
    ctxs = input_data['ctxs']
    input = '\nContext:\n'
    for i, ctx in enumerate(ctxs):
        input += f'Document {i+1}: {ctx}\n'
    input = input + input_data['input']
    if input[-2:] == "\n\n":
        input = input[:-2]
    if input[-1:] == "\n":
        input = input[:-1]
    input_data['input'] = input
    input_data.pop('ctxs')

    return input_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="Path to the input file.", required=True)
    parser.add_argument('--output', type=str, help="Path to the output file.", required=True)

    return parser.parse_args()

def main(args):
    input_data = load_jsonlines(args.input)
    processed_data = []
    for item in input_data:
        item = process_instruction(item)
        item = process_input(item)
        processed_data.append(item)
    save_file_jsonl(processed_data, args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)