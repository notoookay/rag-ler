from datasets import load_dataset

import argparse
import random
import jsonlines

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, help="Path to the output file.", required=True)
    parser.add_argument('--n', type=int, help="Number of samples to take.", required=False)

    return parser.parse_args()

def get_supporting_facts_text(row):
    contexts = row['context']
    supporting_facts_text = []
    for i, title in enumerate(contexts['title']):
        if title in row['supporting_facts']['title']:
            sentences = contexts['sentences'][i]
            text = ""
            for sentence in sentences:
                text += sentence
            supporting_facts_text.append({'title': title, 'text': text})

    return supporting_facts_text

def get_contexts(x):
    ctxs = []
    for ctx in x:
        ctxs.append(f"(Title: {ctx['title']}) {ctx['text']}")
    return ctxs

def main(args):
    input_data = load_dataset('hotpot_qa', 'fullwiki', split='train')
    new_data = []
    for idx, item in enumerate(input_data):
        output = item['answer']
        supporting_facts_text = get_supporting_facts_text(item)
        ctxs = get_contexts(supporting_facts_text)
        input = item['question']
        id = f"hotpotqa_{idx}"
        dataset_name = 'hotpotqa'

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