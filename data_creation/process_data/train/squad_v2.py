from datasets import load_dataset

import argparse
import random
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, help="Path to the output file.", required=True)
    parser.add_argument('--n', type=int, help="Number of samples to take.", required=False)

    return parser.parse_args()

def get_output(x):
    if x['text']:
        return x['text'][0]
    return "[No Answer]"

def main(args):
    input_data = load_dataset('rajpurkar/squad_v2')['train']
    new_data = []
    for idx, item in enumerate(input_data):
        input = item['question']
        output = get_output(item['answers'])
        id = f"squad_{idx}"
        ctxs = [f"(Title: {item['title']}) {item['context']}"]
        dataset_name = 'squad_v2'

        new_data.append({
            "id": id,
            "input": input,
            "output": output,
            "ctxs": ctxs,
            "dataset_name": dataset_name
        })
    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)
    
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)

if __name__ == "__main__":
    args = parse_args()
    main(args)