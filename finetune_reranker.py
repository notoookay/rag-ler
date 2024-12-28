#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import datasets
from datetime import timedelta, datetime
import time
from copy import deepcopy

import torch
from torch.nn import functional as F
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from huggingface_hub import HfApi
import numpy as np
import deepspeed

from utils import push_folder_to_hub
from evaluate_passages import (
    evaluate_rankings,
    convert_ctxs,
    rerank_contexts,
    load_data,
)

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A json file containing the training data."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, limit the number of training examples.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        help="""If given, specifies a model revision (for HuggingFace models). This will 
        be applied to both the `model_name_or_path` and `config_name` args.""",
        default="main",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_revision",
        help="""Specifies a revision for the tokenizer. If not given, defaults
             to the value of the `model_revision` arg. In most cases, the tokenizer
             revision should be the same as the model revision and this flag shouldn't
             be needed.""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=os.path.basename(__file__)[: -len(".py")],
        help="Specify the experiment name for tracking experiment with tools like"
             "`wandb`, `mlflow`, `tensorboard`."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Specify the run name for tracking experiment with tools like"
             "`wandb`, `mlflow`, `tensorboard`."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    parser.add_argument(
        '--ctxs_num',
        type=int,
        default=30,
        help='Number of contexts to consider for the task.',
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        default=True,
        help='Push the model to the hub when training ends.',
    )
    parser.add_argument(
        '--hf_repo_id',
        type=str,
        default=None,
        help='The name of the repo to push to.',
    )
    parser.add_argument(
        '--hf_entity',
        type=str,
        default=None,
        help='The name of the entity to push to.',
    )
    parser.add_argument(
        '--hf_repo_revision',
        type=str,
        default=None,
        help='The revision of the repo to push to.',
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        help="A JSON file containing the evaluation data.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to perform evaluation during training.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Max number of new tokens to generate during evaluation.",
    )
    parser.add_argument(
        "--metrics",
        choices=["em", "accuracy", "f1"],
        nargs="+",
        default=["accuracy"],
        help="Evaluation metrics to compute.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for data loading.",
    )
    parser.add_argument(
        "--use_context",
        action="store_true",
        help="Whether to use context in the prompt.",
    )
    parser.add_argument(
        "--use_chat_prompt",
        action="store_true",
        help="Whether to use chat model.",
    )
    parser.add_argument(
        '--d_conf_threshold',
        type=float,
        default=0.8,
        help='LLM decision confidence threshold',
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args

def save_with_accelerate(accelerator, model, output_dir):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict,
        safe_serialization=False
    )

def tokenize_func(example, tokenizer, max_length):
    input_ids = []
    attention_mask = []
    token_type_ids = None

    # RoBERTa doesn't use token_type_ids
    if not (isinstance(tokenizer, transformers.XLMRobertaTokenizer) or isinstance(tokenizer, transformers.RobertaTokenizer)):
        token_type_ids = []
    input = example['input']
    ctxs = example['ctxs']
    for ctx in ctxs:
        tokenized = tokenizer(input, ctx, truncation=True, max_length=max_length)
        input_ids.append(tokenized['input_ids'])
        attention_mask.append(tokenized['attention_mask'])
        if token_type_ids is not None:
            token_type_ids.append(tokenized['token_type_ids'])

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids

    return inputs

def get_confidence(log_prob_lsr):

    # max(p) / sum(p)
    # prob = np.exp(log_prob_lsr)
    # confidence = max(prob) / sum(prob)

    # entropy
    prob = np.exp(log_prob_lsr)
    entropy = -np.sum(prob * np.log(prob))
    confidence = 1 - entropy / np.log(len(prob))

    return confidence

# Prepare reference model separately
def prepare_deepspeed(accelerator, model):
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like:
                # `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and
    # is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    args.output_dir = os.path.join(
        args.output_dir,
        f"{args.model_name_or_path.replace('/', '_')}__{args.d_conf_threshold}__{args.train_file.split('/')[-1].split('.')[0]}__{args.seed}",
    )
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs]
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load evaluation data
    if args.evaluate:
        eval_data = load_data(args.eval_file)

        k_values = [1, 3, 5, 10, 20]  # Adjust as needed
        convert_ctxs(eval_data)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            torch_dtype='auto',
            return_dict=True,
            trust_remote_code=args.trust_remote_code,
        )
        # Create a copy of the model before training for reference outputs
        reference_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            torch_dtype='auto',
            return_dict=True,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSequenceClassification.from_config(config)
        reference_model = AutoModelForSequenceClassification.from_config(config)

    train_dataset = raw_datasets["train"]

    # debugging tool for fewer samples
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    # tokenize the dataset
    train_dataset = train_dataset.map(
        partial(tokenize_func,
                tokenizer=tokenizer,
                max_length=args.max_seq_length),
        batched=False,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['id', 'input', 'output', 'dataset_name', 'ctxs', 'log_prob_context'],
        # remove_columns=['query_id', 'query', 'retrieved_passages', 'positive_passages', 'log_probs', 'relevance_score'],
        desc="Tokenizing dataset",
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def collate_fn(examples):
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        if examples[0].get('token_type_ids') is not None:
            tokenize_input = {'input_ids': [],
                              'attention_mask': [],
                              'token_type_ids': []}
        else:
            tokenize_input = {'input_ids': [],
                              'attention_mask': []}

        log_prob_lsrs = []
        confidences = []
        for example in examples:
            for key in tokenize_input.keys():
                tokenize_input[key].extend(example[key])
            log_prob_lsrs.append(example['log_prob_lsrs'])
            confidences.append(get_confidence(example['log_prob_lsrs']))
            example.pop('log_prob_lsrs')

        tokenized = tokenizer.pad(
            tokenize_input,
            padding="longest",
            max_length=args.max_seq_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        tokenized['log_prob_lsrs'] = torch.tensor(log_prob_lsrs)
        tokenized['confidences'] = torch.tensor(confidences)

        return tokenized

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Prepare reference model
    reference_model = prepare_deepspeed(accelerator, reference_model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value

        # use wandb for tracking
        args.run_name = f"{args.model_name_or_path.replace('/', '_')}__{args.d_conf_threshold}__{args.train_file.split('/')[-1].split('.')[0]}__{args.seed}__{int(time.time())}"
        accelerator.init_trackers(args.experiment_name,
                                  experiment_config,
                                  init_kwargs={"wandb": {"name": args.run_name}})

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            model.train()
            with accelerator.accumulate(model):
                log_prob_lsrs = batch.pop('log_prob_lsrs')
                confidences = batch.pop('confidences')
                # truncate confidences less than threshold
                confidences[confidences < args.d_conf_threshold] = 0
                alpha = confidences.unsqueeze(1)
                outputs = model(**batch)
                logits = outputs.logits
                logits = logits.view(-1, args.ctxs_num) # (batch_size, ctxs_num)
                with torch.no_grad():
                    reference_outputs = reference_model(**batch)
                reference_logits = reference_outputs.logits
                reference_logits = reference_logits.view(-1, args.ctxs_num)
                log_prob_lsrs = log_prob_lsrs.view(-1, args.ctxs_num)
                log_p_r = torch.log(F.softmax(logits, dim=-1))
                log_p_o = torch.log(F.softmax(reference_logits, dim=-1))
                # use KL Divergence loss
                learning_loss = F.kl_div(log_p_r, log_prob_lsrs, reduction='none', log_target=True)
                reference_loss = F.kl_div(log_p_r, log_p_o, reduction='none', log_target=True)
                loss = (1 - alpha) * reference_loss + alpha * learning_loss
                # Align with batch mean (reduction='batchmean')
                loss = loss.sum() / log_p_r.size(0)
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(accelerator, model, output_dir)

                        if args.evaluate:
                            model.eval()
                            logger.info("Evaluating...")

                            # Copy data to avoid modifying the original
                            eval_data_copy = deepcopy(eval_data)
                            rerank_contexts(eval_data_copy, model, tokenizer)

                            logger.info("Re-evaluating rankings after reranking")
                            reranked_results = evaluate_rankings(eval_data_copy, k_values, logger)
                            # Log reranked results to wandb
                            for k, metrics in reranked_results.items():
                                accelerator.log(
                                    {
                                        "train_step": completed_steps,
                                        f"Reranked MRR@{k}": metrics["MRR@K"],
                                        f"Reranked R@{k}": metrics["R@K"],
                                    },
                                    step=completed_steps,
                                )

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        output_dir = args.output_dir
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
        save_with_accelerate(accelerator, model, output_dir)

    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "rag-ler"
        if args.hf_entity is None:  # try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = (
                args.run_name
            )
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)


if __name__ == "__main__":
    main()
