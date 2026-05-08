# Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning

## Overview

This repository contains the code for the MSc. thesis "Improving Low-Resource
Machine Translation via Round-Trip Reinforcement Learning" together with the
extension experiments for **Maithili (mai_Deva)** using NLLB-200-distilled-600M.

The extension adds an extra fluency component (LMScore) to the round-trip
reward, giving three experiment configurations:

1. **baseline_eval** — vanilla pretrained NLLB-600M, evaluation only.
2. **rl_baseline** — original paper reward `R = chrF++ + BLEU`.
3. **modified_rl** — proposed reward `R = 0.7·chrF++ + 0.2·BLEU + 0.1·LMScore`.

LMScore is the cosine similarity between embeddings of the original English
sentence and the back-translated English sentence, computed with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and mapped to
`[0, 1]`.

## Setup

```bash
pip install -r requirements.txt
```

A single GPU is sufficient for NLLB-600M.

### Colab + AWS S3 setup

The repo supports reading the dataset directly from S3 (`s3://...` paths via
`s3fs`) and pushing all per-experiment outputs to S3 (`boto3`). On each Colab
session run:

```bash
git clone <REPO_URL> && cd MT-via-Round-Trip-RL
pip install -q -r requirements.txt
```

Then export AWS credentials in the notebook *before* `python main.py ...`:

```python
import os
os.environ["AWS_ACCESS_KEY_ID"]     = "..."   # or read from Colab secrets
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_DEFAULT_REGION"]    = "us-east-1"
```

Outputs land under `{s3_output_uri}/{experiment_name}/{results,plots,model}`,
so three concurrent Colab sessions writing to distinct `experiment_name`s never
collide on S3.

## Dataset format

The Maithili task expects English↔Maithili parallel data. Either

- use a Hugging Face dataset (set `task.data.path` and
  `task.data.dataset_config_name`), or
- pass local CSV / TSV / JSONL files **or `s3://...` URIs** with the columns
  `sentence_eng_Latn` and `sentence_mai_Deva` via:

```bash
task.data.train_file=s3://my-bucket/maithili/train.csv
task.data.valid_file=s3://my-bucket/maithili/valid.csv
task.data.test_file=s3://my-bucket/maithili/test.csv
```

Recommended sizes: train 5–15k, valid 500, test 500.

## Training & evaluation commands

All commands use the `nllb_maithili` task config. Override fields with the
standard Hydra `key=value` syntax.

Replace `S3_BUCKET` with your bucket and `S3_PREFIX` with a folder
(e.g. `mt-rl/runs`). Each command is meant for one Colab session; the three
can run concurrently because each writes to its own `name=` namespace.

### 1. Baseline (no fine-tuning, evaluate only)

```bash
python main.py \
    task=nllb_maithili \
    task.experiment.mode=baseline_eval \
    task.experiment.name=baseline_eval_maithili \
    task.experiment.s3_output_uri=s3://S3_BUCKET/S3_PREFIX \
    task.data.train_file=s3://S3_BUCKET/maithili/train.csv \
    task.data.valid_file=s3://S3_BUCKET/maithili/valid.csv \
    task.data.test_file=s3://S3_BUCKET/maithili/test.csv
```

### 2. RL baseline (chrF++ + BLEU reward)

```bash
python main.py \
    task=nllb_maithili \
    task.experiment.mode=rl_baseline \
    task.experiment.name=rl_baseline_maithili \
    task.experiment.s3_output_uri=s3://S3_BUCKET/S3_PREFIX \
    task.experiment.upload_model_to_s3=true \
    task.data.train_file=s3://S3_BUCKET/maithili/train.csv \
    task.data.valid_file=s3://S3_BUCKET/maithili/valid.csv \
    task.data.test_file=s3://S3_BUCKET/maithili/test.csv
```

### 3. Modified RL (chrF++ + BLEU + LMScore)

```bash
python main.py \
    task=nllb_maithili \
    task.experiment.mode=modified_rl \
    task.experiment.name=modified_rl_maithili \
    task.experiment.s3_output_uri=s3://S3_BUCKET/S3_PREFIX \
    task.experiment.upload_model_to_s3=true \
    task.data.train_file=s3://S3_BUCKET/maithili/train.csv \
    task.data.valid_file=s3://S3_BUCKET/maithili/valid.csv \
    task.data.test_file=s3://S3_BUCKET/maithili/test.csv
```

## Outputs

Each run writes locally:

- `results/{experiment_name}.json` — full metric history + final eval/test
  numbers (BLEU, chrF++, TER, BERTScore).
- `results/training_log_{experiment_name}.csv` — per-eval row
  (`step, reward, bleu, chrf++, ter, bertscore`).
- `results/summary.csv` — appended one row per run (local-only, paper-table
  friendly; not synced to S3 to avoid races between concurrent runs).
- `plots/{experiment_name}/` — `reward_vs_steps.png`, `chrf_vs_steps.png`,
  `bleu_vs_steps.png`, `ter_vs_steps.png`, `bertscore_vs_steps.png`.

When `task.experiment.s3_output_uri` is set, the same files are mirrored to
`{s3_output_uri}/{experiment_name}/results/` and
`{s3_output_uri}/{experiment_name}/plots/{experiment_name}/`. Model
checkpoints are uploaded only when `task.experiment.upload_model_to_s3=true`.

After all three runs finish, you can rebuild the consolidated paper table by
downloading the three per-experiment JSONs:

```bash
mkdir -p results
aws s3 cp s3://S3_BUCKET/S3_PREFIX/baseline_eval_maithili/results/baseline_eval_maithili.json results/
aws s3 cp s3://S3_BUCKET/S3_PREFIX/rl_baseline_maithili/results/rl_baseline_maithili.json results/
aws s3 cp s3://S3_BUCKET/S3_PREFIX/modified_rl_maithili/results/modified_rl_maithili.json results/
```

## Qualitative examples

After running all three experiments, pull the two trained checkpoints down
from S3 (the `baseline` model is just `facebook/nllb-200-distilled-600M`):

```bash
aws s3 cp --recursive \
    s3://S3_BUCKET/S3_PREFIX/rl_baseline_maithili/model \
    ./ckpts/rl_baseline_maithili
aws s3 cp --recursive \
    s3://S3_BUCKET/S3_PREFIX/modified_rl_maithili/model \
    ./ckpts/modified_rl_maithili
```

Then generate side-by-side translations:

```bash
python generate_examples.py \
    --baseline_model facebook/nllb-200-distilled-600M \
    --rl_baseline_model ./ckpts/rl_baseline_maithili \
    --modified_rl_model ./ckpts/modified_rl_maithili \
    --test_file data/mai_test.csv \
    --source_lang eng_Latn --target_lang mai_Deva \
    --num_examples 20 \
    --output qualitative_examples.csv
```

## Reproducibility

`configs/train.yaml` sets `seed: 27`. The training script seeds Python,
NumPy and PyTorch and toggles cuDNN to deterministic mode. Note that some
CUDA kernels remain non-deterministic in mixed precision; results are
expected to be reproducible to within a small tolerance.

## Key files

- `main.py` — training / evaluation loop, plot + metrics export.
- `utils.py` — GRPO loss, `LMScorer`, `compute_translation_metrics`.
- `dl.py` — dataset loading (HF dataset + local CSV/TSV/JSONL fallback).
- `generate_examples.py` — qualitative outputs across the three modes.
- `configs/task/nllb_maithili.yaml` — Maithili task config with experiment block.
- `baselines/` — original UMNMT and back-translation baselines.

## Citation

```bibtex
@misc{attia2026improvinglowresourcemachinetranslation,
      title={Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning},
      author={Ahmed Attia and Alham Fikri Aji},
      year={2026},
      eprint={2601.12535},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.12535},
}
```
