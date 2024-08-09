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
                        default="wg",)
    args = parser.parse_args()
    
    data = datasets.load_dataset("coref-data/winogrande_raw", "winogrande_debiased")["validation"]
    new_data = []
    for i, item in enumerate(data):
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]
        answer = item["answer"]
        q_id = "{0}_{1}".format(args.data_prefix, i)
        input_question = "{0}\nA: {1}\nB: {2}".format(sentence, option1, option2)
        output = "A" if answer == "1" else "B"
        new_data.append({"output": output, "input": input_question, "id": q_id, "dataset_name": args.data_prefix})

    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)
    
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)

if __name__ == "__main__":
    main()
