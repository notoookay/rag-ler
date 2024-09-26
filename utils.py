import jsonlines
import json
import copy
import re
import time

from accelerate import Accelerator
from typing import Optional
from huggingface_hub import HfApi

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
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

TASK_INST = {"hotpotqa": "Answer the following question.",
             "newsqa": "Answer the following question.",
             "squad_v2": "Answer the following question.",
             "race": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "csqa": "Given five answer candidates, A, B, C, D and E, choose the best answer choice.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
             "trivia_qa": "Answer the following question.",
             "nq": "Answer the following question.",
             "fever": "Is the following statement correct or not? Say 'true' if it's correct; otherwise say 'false'.",
             "boolq": "Given a passage and a yes/no question about that passage, determine if the correct answer is 'yes' or 'no' based solely on the information provided in the passage.",
             "wg": "Given a sentence with a blank and two options, A and B, choose the option that best fits the blank.",
             "piqa": "Given a goal that describes a physical task or situation, choose the most appropriate solution from two options A and B."}

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

NO_ANSWER_TOKEN = "[No Answer]"
SPECIAL_TOKENS = {"no_answer_token": NO_ANSWER_TOKEN}


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


def preprocess_input(input_data, task):
    if task == "qa":
        for item in input_data:
            if "instruction" not in item:
                item["instruction"] = item["question"]
            if "answers" not in item and "output" in item:
                item["answers"] = "output"
        return input_data

    elif task in ["asqa", "eli5"]:
        processed_input_data = []
        for instance_idx, item in enumerate(input_data["data"]):
            prompt = item["question"]
            instructions = TASK_INST[task]
            prompt = instructions + "## Input:\n\n" + prompt
            entry = copy.deepcopy(item)
            entry["instruction"] = prompt
            processed_input_data.append(entry)
        return processed_input_data


def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key in ["A", "B", "C", "D"]:
            answer_labels[answer_key] = text

    if "D" not in answer_labels:
        answer_labels["D"] = ""
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
    return processed_instruction


def retry_on_exception(max_attempts=4, delay=1, backoff=2):
    """
    Retry a function on exception. Useful for HF API calls that may fail due to
    network issues. E.g., https://beaker.org/ex/01J69P87HJQQ7X5DXE1CPWF974
    `huggingface_hub.utils._errors.HfHubHTTPError: 429 Client Error`

    We can test it with the following code.
    @retry_on_exception(max_attempts=4, delay=1, backoff=2)
    def test():
        raise Exception("Test exception")

    test()
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            local_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed. Retrying in {local_delay} seconds...")
                    time.sleep(local_delay)
                    local_delay *= backoff
            return None

        return wrapper

    return decorator

@retry_on_exception()
def push_folder_to_hub(
    accelerator: Accelerator,
    output_dir: str,
    hf_repo_id: Optional[str] = None,
    hf_repo_revision: Optional[str] = None,
    private: bool = True,
):
    if accelerator.is_main_process:
        hf_repo_url = f"https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}"
        api = HfApi()
        if not api.repo_exists(hf_repo_id):
            api.create_repo(hf_repo_id, exist_ok=True, private=private)
        if hf_repo_revision is not None:
            api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
        api.upload_folder(
            repo_id=hf_repo_id,
            revision=hf_repo_revision,
            folder_path=output_dir,
            commit_message="upload checkpoint",
            run_as_future=False,
        )
        print(f"🔥 pushed to {hf_repo_url}")