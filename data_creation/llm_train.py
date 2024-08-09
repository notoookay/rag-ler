""" Get the training data for the LLM """

# We expect the input data following the format:
# (id, input, output, ctxs, dataset_name, instruction)

from tqdm import tqdm

import argparse

from utils import load_jsonlines, save_file_jsonl, TASK_INST_CONTEXT

INPUT_FORMAT = {
    "context": "Context:\n{ctxs}\nQuestion:\n{input}",
    "no_context": "Question:\n{input}",
}

def format_input(input, ctxs=None):
    if ctxs is None or len(ctxs) == 0:
        return INPUT_FORMAT["no_context"].format(input=input)
    
    ctxs = "\n".join([f"Document {i+1}: {ctx}" for i, ctx in enumerate(ctxs)])    
    
    return INPUT_FORMAT["context"].format(ctxs=ctxs, input=input)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()

def main(args):
    input_data = load_jsonlines(args.input)
    output_data = []
    for item in tqdm(input_data, desc="Processing data"):
        id = item["id"]
        dataset_name = item['dataset_name']
        # for QA tasks, we need to format the input
        if dataset_name in TASK_INST_CONTEXT:
            input = format_input(item["input"], item["ctxs"])
            instruction = TASK_INST_CONTEXT[dataset_name]
        # for instruction following tasks, we just need to get the input
        else:
            input = item['input']
            instruction = item['instruction']
        output = item["output"]
        output_data.append({
            "id": id,
            "input": input,
            "output": output,
            "instruction": instruction,
            "dataset_name": dataset_name
        })

    print(f"Saving data to {args.output}")
    save_file_jsonl(output_data, args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)
