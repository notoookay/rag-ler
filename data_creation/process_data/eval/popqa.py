from datasets import load_dataset

import argparse
import jsonlines
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default=None)
    parser.add_argument('--data_prefix', type=str,
                        default='popqa')
    args = parser.parse_args()

    data = load_dataset('akariasai/PopQA', split='test')
    # get long tail items
    data = data.filter(lambda x: x['s_pop'] < 100)

    new_data = []
    for item in data:
        question = item['question']
        answer = eval(item['possible_answers'])  # Convert string to list
        q_id = f'{args.data_prefix}_{item["id"]}'
        input_question = f'{question}'
        output = answer
        new_data.append({'output': output, 'input': input_question, 'id': q_id, 'dataset_name': args.data_prefix})

    output_file = os.path.join(args.output_dir, f'{args.data_prefix}_longtail.jsonl')
    print(f"Saving the processed data to {output_file}...")
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(new_data)

if __name__ == "__main__":
    main()