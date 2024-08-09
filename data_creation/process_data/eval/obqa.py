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
                        default="obqa",)
    args = parser.parse_args()
    
    data = datasets.load_dataset("allenai/openbookqa", "additional")["validation"]
    new_data = []
    for item in data:
        question_stem = item["question_stem"]
        choices = item["choices"]
        choices["label"] = [ "A", "B", "C", "D" ]
        answer_key = item["answerKey"]
        q_id = "{0}_{1}".format(args.data_prefix, item["id"])
        answer_labels = {}
        for text, choice in zip(choices["text"], choices["label"]):
            answer_labels[choice] = text
        input_question = "{0}\nA: {1}\nB: {2}\nC: {3}\nD: {4}".format(question_stem, answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
        output = answer_key
        new_data.append({"output": output, "input": input_question, "id": q_id, "dataset_name": args.data_prefix})

    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)
    
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)

if __name__ == "__main__":
    main()
