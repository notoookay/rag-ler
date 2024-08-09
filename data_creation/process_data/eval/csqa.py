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
                        default="csqa")
    args = parser.parse_args()

    data = datasets.load_dataset("tau/commonsense_qa")["validation"]
    new_data = []
    for item in data:
        question_stem = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]
        if answer_key == "1":
            answer_key = "A"
        if answer_key == "2":
            answer_key = "B"
        if answer_key == "3":
            answer_key = "C"
        if answer_key == "4":
            answer_key = "D"
        if answer_key == "5":
            answer_key = "E"
        answer_labels = {}
        choices["label"] = ["A", "B", "C", "D", "E"]
        if len(choices["text"]) < 4:
            continue
        for text, choice in zip(choices["text"], choices["label"]):
            answer_labels[choice] = text
        q_id = "{0}_{1}".format(args.data_prefix, item["id"])
        input_question = "{0}\nA: {1}\nB: {2}\nC: {3}\nD: {4}\nE: {5}".format(
            question_stem,
            answer_labels["A"],
            answer_labels["B"],
            answer_labels["C"],
            answer_labels["D"],
            answer_labels["E"],
        )
        output = answer_key
        new_data.append({"output": output, "input": input_question, "id": q_id, "dataset_name": args.data_prefix})

    if args.n is not None:
        new_data = random.sample(new_data, k=args.n)

    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(new_data)

if __name__ == "__main__":
    main()
