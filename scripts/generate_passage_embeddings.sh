python retrieval/generate_passage_embeddings.py \
    --model_name_or_path facebook/contriever-msmarco \
    --output_dir contriever_embeddings  \
    --passages psgs_w100.tsv \
    --shard_id 0 --num_shards 1 \