from datasets import load_dataset

import argparse
import random
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="Path to the input file.", required=True)
    parser.add_argument('--output_file', type=str, help="Path to the output file.", required=True)
    parser.add_argument('--n', type=int, help="Number of samples to take.", required=False)

    return parser.parse_args()

def main(args):
    input_data = load_dataset('json', data_files=args.input_file)['train']
    new_data = []
    for idx, item in enumerate(input_data):
        # only take the first answer, you can also randomly sample from the answers
        output = item['output'][0]['answer']
        input = item['input']
        id = f"nq_{idx}"
        dataset_name = 'nq'

        new_data.append({
            "id": id,
            "input": input,
            "output": output,
            "dataset_name": dataset_name
        })
    
    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)
    
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)