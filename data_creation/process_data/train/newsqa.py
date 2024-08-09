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

def get_answers(row):
    result = list()
    ranges = row['answer_token_ranges']
    for span in ranges.split(','):
        if span == 'None':
            continue
        s, e = map(int, span.split(':'))
        tokens = row['story_text'].split()[s:e]
        ans = ""
        for token in tokens:
            ans += token + " "
        ans = ans.strip()
        result.append(ans)
    return result

def main(args):
    input_data = load_dataset('newsqa', data_dir=args.input_file)['train']
    new_data = []
    for idx, item in enumerate(input_data):
        output = get_answers(item)[0]
        ctxs = item['story_text']
        input = item['question']
        id = f"newsqa_{idx}"
        ctxs = [ctxs]
        dataset_name = 'newsqa'

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