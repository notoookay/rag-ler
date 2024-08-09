echo "Downloading the flan_v2 chain-of-thought submix..."
wget -P data/raw_train/cot/ https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/cot_zsopt.jsonl
wget -P data/raw_train/cot/ https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/cot_fsopt.jsonl

echo "Downloading the flan_v2 collection, here we use two subsampled versions: for tulu v1 we subsampled 100K, for tulu v2 we subsampled 50K..."
mkdir -p data/raw_train/flan_v2/
wget -O data/raw_train/flan_v2/tulu_v2_resampled_flan_50k.jsonl https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/flan_v2_resampled_50k.jsonl

echo "Downloading the OpenAssistant data (oasst1)..."
wget -P data/raw_train/oasst1/ https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz
gzip -d data/raw_train/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz

echo "Downloading the gpt4-llm dataset..."
wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json

echo "Downloading ShareGPT dataset..."
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json
echo "Splitting the ShareGPT dataset with 2048 max tokens per conversation..."
python process_data/split_sharegpt_conversations.py \
    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048.json \
    --model-name-or-path oobabooga/llama-tokenizer \
    --max-length 2048

echo "Processing tulu datasets..."
python process_data/train/reformat_tulu_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/
