import jsonlines
import json
import re

TASK_INST_CONTEXT = {"hotpotqa": "Answer the following question based on the provided contexts. You may use one or more provided contexts.",
             "newsqa": "Answer the following question based on the provided contexts. You may use one or more provided contexts.",
             "squad_v2": "Answer the following question based on the provided contexts. You may use one or more provided contexts.",
             "race": "Given four answer candidates, A, B, C and D, choose the best answer choice. You may use one or more provided contexts.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice. You may use one or more provided contexts.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice. You may use one or more provided contexts.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice. You may use one or more provided contexts.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
             "trivia_qa": "Answer the following question based on the provided contexts. You may use one or more provided contexts.",
             "nq": "Answer the following question based on the provided contexts. You may use one or more provided contexts.",
             "fever": "Is the following statement correct or not? Say 'true' if it's correct; otherwise say 'false'. You may use one or more provided contexts.",}

TASK_INPUT = {
    "hotpotqa": "Question",
    "newsqa": "Question",
    "squad_v2": "Question",
    "race": "Question",
    "obqa": "Question",
    "arc_easy": "Question",
    "arc_c": "Question",
    "csqa": "Question",
    "asqa": "Question",
    "trivia_qa": "Question",
    "nq": "Question",
    "fever": "Statement",
    "boolq": "Question",
    "wg": "Sentence",
    "piqa": "Goal",
}

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": ("### Instruction:\n{instruction}\n\n### Response:\n"),
}

PROMPT = "{shots}{task_input}:\n{input}\n\nResponse:\n"
PROMPT_CONTEXT = "{shots}Context:\n{context}\n\n{task_input}:\n{input}\n\nResponse:\n"
SHOT_TEMPLATE = "{task_input}:\n{input}\n\nResponse:\n{output}\n\n"
SHOT_TEMPLATE_CONTEXT = "Context:\n{context}\n\n{task_input}:\n{input}\n\nResponse:\n{output}\n\n"

def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)
