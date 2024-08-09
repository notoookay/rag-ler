from datasets import load_dataset

import argparse
import random
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, help="Path to the output file.", required=True)
    parser.add_argument('--n', type=int, help="Number of samples to take.", required=False)

    return parser.parse_args()

def convert_input(x):
    question = x['question']
    options = x['options']
    input_question = "{0}\nA: {1}\nB: {2}\nC: {3}\nD: {4}".format(question, options[0], options[1], options[2], options[3])
    
    return input_question

def main(args):
    input_data = load_dataset('ehovy/race', 'all')['train']
    new_data = []
    for idx, item in enumerate(input_data):
        input = convert_input(item)
        output = item['answer']
        id = f"race_{idx}"
        ctxs = [item['article']]
        dataset_name = 'race'

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