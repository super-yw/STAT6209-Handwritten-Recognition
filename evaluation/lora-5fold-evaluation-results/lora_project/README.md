# Qwen3-VL-4B-Instruct LoRA Project

This project fine-tunes `Qwen3-VL-4B-Instruct` for the handwritten-number extraction task in this repository.

The training target matches the dataset prompt:

- Return the handwritten number when present.
- If no handwritten number exists, return the most prominent printed number.
- If there is only a crossed-out mark and no handwritten number, return `0`.
- Return only the number string.

## Base Model Safety

The base checkpoint is read from:

- `/data/Work/Quark/models/Qwen3-VL-4B-Instruct`

This project never writes back into that directory. LoRA adapters and training artifacts are saved separately under `/data`.

## Data Inputs

The project uses the processed manifests under:

- `training&validate data for lora/processed/fold_<N>/train_metadata.csv`
- `training&validate data for lora/processed/fold_<N>/eval_metadata.csv`

If `eval_metadata.csv` is missing, the code can rebuild the eval split from `dataset_summary.json` and `complete_dataset_metadata.csv`.

## Install

```bash
cd /home/i-zhaowei/STAT6209-Handwritten-Recognition/lora_project
python3 -m pip install -r requirements.txt
```

Optional for QLoRA:

```bash
python3 -m pip install bitsandbytes
```

## Train One Fold

```bash
cd /home/i-zhaowei/STAT6209-Handwritten-Recognition/lora_project
bash scripts/run_fold.sh 1
```

Default outputs go to:

- `/data/Work/Quark/lora_outputs/stat6209_qwen3vl4b_instruct/fold_1`

This training flow now does three kinds of evaluation automatically:

- run the base model on the fold eval set before LoRA training starts
- run generation-based eval after every epoch
- run generation-based eval again on the final best adapter

## Train All 5 Folds

```bash
cd /home/i-zhaowei/STAT6209-Handwritten-Recognition/lora_project
bash scripts/train_all_folds.sh
```

## Evaluate One Fold

```bash
cd /home/i-zhaowei/STAT6209-Handwritten-Recognition/lora_project
bash scripts/eval_fold.sh 1
```

This writes predictions and exact-match metrics to:

- `/data/Work/Quark/lora_outputs/stat6209_qwen3vl4b_instruct/fold_1/eval_predictions_fold_1.json`

## Evaluate All 5 Folds

Base model only:

```bash
cd /home/i-zhaowei/STAT6209-Handwritten-Recognition/lora_project
bash scripts/eval_all_folds_base.sh
```

LoRA adapters only:

```bash
cd /home/i-zhaowei/STAT6209-Handwritten-Recognition/lora_project
bash scripts/eval_all_folds_adapter.sh
```

## Summarize 5-Fold Results

```bash
cd /home/i-zhaowei/STAT6209-Handwritten-Recognition/lora_project
bash scripts/summarize_five_folds.sh
```

This writes:

- `/data/Work/Quark/lora_outputs/stat6209_qwen3vl4b_instruct/five_fold_summary.json`

## Included Adapters

The repository now also includes the final exported LoRA adapters for all 5 folds under:

- `lora_project/adapters/stat6209_qwen3vl4b_instruct/fold_<N>/adapter_config.json`
- `lora_project/adapters/stat6209_qwen3vl4b_instruct/fold_<N>/adapter_model.safetensors`

Only the final inference adapters are committed here. Intermediate `checkpoint-*` directories, optimizer states, and other training-only artifacts remain outside the repository.

## Notes

- The default training setup is conservative for a small dataset: LoRA rank 8 on `q_proj,v_proj`, batch size 1, grad accumulation 8, 5 epochs, bf16.
- If GPU memory is tight, rerun with `--load_in_4bit true` after installing `bitsandbytes`.
- Exact-match evaluation is handled by `src/stat6209_lora/evaluate.py`; training uses eval loss during fine-tuning.
