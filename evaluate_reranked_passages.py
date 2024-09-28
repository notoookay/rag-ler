import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
import json
import glob
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the data file")

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

def mean_reciprocal_rank_at_k(query_relevances, k):
    """Calculate Mean Reciprocal Rank at K (MRR@K)."""
    reciprocal_ranks = []
    for relevances in query_relevances.values():
        relevances_at_k = relevances[:k]
        try:
            rank = relevances_at_k.index(1) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0)
    
    return np.mean(reciprocal_ranks)

def recall_at_k(query_relevances, k):
    """Calculate Recall at K."""
    recalls = []
    for relevances in query_relevances.values():
        total_relevant = sum(relevances)
        if total_relevant == 0:
            recalls.append(1.0)  # If no relevant items, consider it perfect recall
        else:
            relevant_at_k = sum(relevances[:k])
            recalls.append(relevant_at_k / total_relevant)
    
    return np.mean(recalls)

def evaluate_rankings(data, k_values, logger):
    """Evaluate rankings using MRR@K and Recall@K for multiple K values."""
    query_relevances = defaultdict(list)
    
    for item in data:
        query_id = item['id']
        passages = list(item['ctxs'].values())
        # Use the index of each passage as its rank
        relevances = [1 if passage['hasanswer'] else 0 for passage in passages]
        query_relevances[query_id] = relevances

    results = {}
    for k in k_values:
        mrr_k = mean_reciprocal_rank_at_k(query_relevances, k)
        recall_k = recall_at_k(query_relevances, k)
        results[k] = {
            'MRR@K': mrr_k,
            'Recall@K': recall_k
        }
        logger.info(f"K = {k}")
        logger.info(f"  MRR@{k}: {mrr_k:.3f}")
        logger.info(f"  Recall@{k}: {recall_k:.3f}")
    
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
    return data

def main(args):
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"ranking_evaluation_{timestamp}.log"
    logger = setup_logger(log_file)

    # Load data
    data_paths = glob.glob(args.data)
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

        k_values = [5, 10, 20]  # Adjust as needed

        logger.info("Starting ranking evaluation")
        logger.info(f"Number of queries: {len(data)}")
        logger.info(f"K values being evaluated: {k_values}")

        results = evaluate_rankings(data, k_values, logger)

    logger.info("Evaluation complete")

if __name__ == "__main__":
    args = parse_args()
    main(args)