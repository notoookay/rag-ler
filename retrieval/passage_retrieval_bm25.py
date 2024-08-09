import os
import argparse
import json
import time
import glob

from pyserini.search.lucene import LuceneSearcher

from src.evaluation import calculate_matches
from src.data import load_passages

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

def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits

def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]

def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        id = d["id"]
        hits = top_passages_and_scores[id]
        docs = [passages[hit.docid] for hit in hits]
        scores = [str(hit.score) for hit in hits]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": hits[c].docid,
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]

def search_passages(searcher, queries, qids, k=100, threads=32):
    start_time_retrieval = time.time()
    hits = searcher.batch_search(
            queries=queries,
            qids=qids,
            k=k,
            threads=threads,
        )
    print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
    
    return hits


def main(args):
    data_paths = glob.glob(args.data)

    # load passages
    print("loading passages")
    passages = load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}
    print("passages have been loaded")

    searcher = LuceneSearcher(args.passages_indexes)
    for path in data_paths:
        data = load_data(path)
        output_path = os.path.join(args.output_dir, os.path.basename(path))

        queries = [ex["input"] for ex in data]  # original is `question`
        qids = [ex["id"] for ex in data]
        hits = search_passages(searcher, queries, qids, args.n_docs, args.threads)
        add_passages(data, passage_id_map, hits)
        if args.validate_retrieval:
            hasanswer = validate(data, args.validation_workers)
            add_hasanswer(data, hasanswer)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as fout:
            for ex in data:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_indexes", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads to use for retrieval")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--validate_retrieval", action="store_true", help="validate retrieval results")

    args = parser.parse_args()
    main(args)
