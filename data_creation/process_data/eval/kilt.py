import argparse

from ...utils import load_jsonlines, save_file_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str,
                        default=None, )
    parser.add_argument('--data_prefix', type=str,
                        default=None,
                        help="dataset name, refer to task instruction keys.")
    return parser.parse_args()

def main(args):
    
    # load jsonlines file
    if args.input_file is None:
        raise ValueError("Please provide an input file to process.")
    input_data = load_jsonlines(args.input_file)

    if args.output_file is None:
        raise ValueError("Please provide an output file to save the processed data.")
    
    new_data = []
    for idx, item in enumerate(input_data):
        id = item['id']
        input = item['input']
        new_item = {"input": input, "id": id, "dataset_name": args.data_prefix}
        # kilt test does not have output field
        if "output" in item:
            outputs = []
            for output in item['output']:
                if 'answer' in output:
                    answer = output['answer']
                    if args.data_prefix == "fever":
                        if answer not in ["REFUTES", "SUPPORTS"]:
                            print(answer)
                        answer = "false" if answer == "REFUTES" else "true"
                    outputs.append(answer)
            new_item['output'] = outputs
        new_data.append(new_item)
    
    print(f"Saving the processed data to {args.output_file}...")
    save_file_jsonl(new_data, args.output_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)
