import argparse
import random
import jsonlines
import datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--output_file', type=str,
                        default=None)
    parser.add_argument('--data_prefix', type=str,
                        default="boolq",)
    args = parser.parse_args()
    
    data = datasets.load_dataset("google/boolq")["validation"]
    new_data = []
    for i, item in enumerate(data):
        q_id = "{0}_{1}".format(args.data_prefix, i)
        output = "yes" if item["answer"] else "no"
        input_question = "{0}\nPassage: {1}".format(item["question"], item["passage"])
        new_data.append({"output": output, "input": input_question, "id": q_id, "dataset_name": args.data_prefix})

    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)
    
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)

if __name__ == "__main__":
    main()
