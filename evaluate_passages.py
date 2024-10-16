import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
import json
import glob
from argparse import ArgumentParser
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from functools import partial
import torch
from tqdm import tqdm

from utils import save_file_jsonl

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the data files")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Whether to rerank passages using reranker, this will keep the evaluation of both the original and reranked passages",
    )
    parser.add_argument(
        "--rerank_model",
        type=str,
        help="Path to the reranker model if reranking is enabled",
    )
    parser.add_argument(
        "--rerank_tokenizer",
        type=str,
        help="set True when rerank is True"
    )
    parser.add_argument(
        "--reranker_revision",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
        default="main"
    )
    parser.add_argument(
        "--ctxs_num",
        type=int,
        default=0,
        help="Number of contexts to use for evaluation. If 0, use all contexts."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the evaluation results",
    )

    return parser.parse_args()

def setup_logger(log_file):
    """Set up logger to write to file and console."""
    logger = logging.getLogger('ranking_evaluation')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def convert_ctxs(data):
    for item in tqdm(data, desc="Converting contexts"):
        ctxs = list(item['ctxs'])
        transformed_ctx = [f"(Title: {ctx['title']}) {ctx['text']}" for ctx in ctxs]
        item['transformed_ctxs'] = transformed_ctx

    return data

def rerank_contexts(data, model, tokenizer):

    for item in tqdm(data, desc="Reranking"):
        inputs = [item['input']] * len(item['transformed_ctxs'])
        # Align with how model trained, use transformed_ctxs as re-ranking passages
        inputs = tokenizer(inputs, item['transformed_ctxs'], padding=True, truncation=True, return_tensors='pt').to(model.device)

        model.eval()
        with torch.no_grad():
            scores = model(**inputs).logits

        scores = scores.squeeze()
        sorted_scores, indices = torch.sort(scores, descending=True)
        sorted_ctxs = [item['ctxs'][i] for i in indices]
        sorted_transformed_ctxs = [item['transformed_ctxs'][i] for i in indices]

        item['ctxs'] = sorted_ctxs
        item['transformed_ctxs'] = sorted_transformed_ctxs
    
    return data

def mean_reciprocal_rank_at_k(query_relevances, k):
    """Calculate Mean Reciprocal Rank at K (MRR@K)."""
    reciprocal_ranks = []
    for relevances in query_relevances:
        relevances_at_k = relevances[:k]
        try:
            rank = relevances_at_k.index(1) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0)

    return np.mean(reciprocal_ranks)

def top_k_hits_recall_at_k(query_relevances, k):
    """Calculate Recall at K."""
    recalls = []
    for relevances in query_relevances:
        total_relevant = sum(relevances)
        if total_relevant == 0:
            recalls.append(0.0)
        else:
            relevant_at_k = 1 if sum(relevances[:k]) else 0
            recalls.append(relevant_at_k)

    return np.mean(recalls)

def evaluate_rankings(data, k_values, logger):
    """Evaluate rankings using MRR@K and R@K for multiple K values."""
    query_relevances = list()
    
    for item in data:
        passages = list(item['ctxs'])
        # Use the index of each passage as its rank
        relevances = [1 if passage['hasanswer'] else 0 for passage in passages]
        query_relevances.append(relevances)

    results = {}
    for k in k_values:
        mrr_k = mean_reciprocal_rank_at_k(query_relevances, k)
        recall_k = top_k_hits_recall_at_k(query_relevances, k)
        results[k] = {
            'MRR@K': mrr_k,
            'R@K': recall_k
        }
        logger.info(f"K = {k}")
        logger.info(f"  MRR@{k}: {mrr_k:.3f}")
        logger.info(f"  R@{k}: {recall_k:.3f}")
    
    return results

def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    else:
        raise ValueError("Data file must be in JSON or JSONL format")
    
    return data

def main(args):
    # Set up logging
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    log_file = f"{args.output}/ranking_evaluation_{args.rerank_model.replace('/', '_')}_{args.reranker_revision}_{args.ctxs_num}.log"
    logger = setup_logger(log_file)

    # Load data
    data_paths = glob.glob(args.data)
    logger.info(f"Number of data files: {len(data_paths)}")
    logger.info(f"Using re-ranker model: {args.rerank_model}_{args.reranker_revision}")
    for data_file in data_paths:
        logger.info(f"Data file: {data_file}")
        try:
            data = load_data(data_file)
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_file}")
            return
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in data file: {data_file}")
            return

        k_values = [1, 3, 5, 10, 20]  # Adjust as needed

        logger.info("Starting ranking evaluation")
        logger.info(f"Number of queries: {len(data)}")
        logger.info(f"K values being evaluated: {k_values}")

        if args.ctxs_num:
            logger.info(f"Using only the first {args.ctxs_num} contexts for evaluation")
            for item in data:
                item['ctxs'] = item['ctxs'][:args.ctxs_num]

        results = evaluate_rankings(data, k_values, logger)

        if args.rerank:
            logger.info("Reranking passages")
            rerank_model = AutoModelForSequenceClassification.from_pretrained(
                args.rerank_model,
                torch_dtype='auto',
                revision=args.reranker_revision).to('cuda')
            if not args.rerank_tokenizer:
                args.rerank_tokenizer = args.rerank_model
            rerank_tokenizer = AutoTokenizer.from_pretrained(
                args.rerank_tokenizer,
                revision=args.reranker_revision,
            )
            convert_ctxs(data)
            rerank_contexts(data, rerank_model, rerank_tokenizer)

            logger.info("Re-evaluating rankings after reranking")
            reranked_results = evaluate_rankings(data, k_values, logger)

            # store the reranked data
            logger.info("Saving reranked data")

            # remove transformed_ctxs before saving
            for item in data:
                del item['transformed_ctxs']

            output_file = f"{args.output}/{args.rerank_model.replace('/', '_')}_{args.reranker_revision}_reranked/"
            if not os.path.exists(output_file):
                os.makedirs(output_file, exist_ok=True)
            save_file_jsonl(
                data,
                output_file + f"{data_file.split('/')[-1]}"
            )

    logger.info("Evaluation complete")

if __name__ == "__main__":
    args = parse_args()
    main(args)
