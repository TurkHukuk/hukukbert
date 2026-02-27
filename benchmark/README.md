# HukukBERT Benchmark

This benchmark is a **cloze test** for Turkish legal language modeling.

Each item provides:
- A legal sentence with exactly one `[MASK]` token.
- A candidate option list.
- A gold (correct) option.

Dataset file: `benchmark/data/hukukbert_v1_cloze.json`  
Total items: **750**

## How to Use

The dataset is a cloze benchmark where each example contains:
- `id`
- `sentence` (includes exactly one `[MASK]`)
- `options` (candidate answers)
- `gold` (correct option)
- `metadata`

The benchmark script is:
- `benchmark/scripts/cloze_benchmark.test.py`

This script expects **JSONL** input (one JSON object per line).  
Convert the dataset first:

```bash
jq -c . benchmark/data/hukukbert_v1_cloze.json > benchmark/data/hukukbert_v1_cloze.jsonl
```

Run one checkpoint:

```bash
python benchmark/scripts/cloze_benchmark.test.py \
  --data-path benchmark/data/hukukbert_v1_cloze.jsonl \
  --models /workspace/models/hukukbert-base-48k-cased/checkpoint-30500 \
  --topk 3 \
  --device auto
```

Run multiple checkpoints and save JSON output:

```bash
python benchmark/scripts/cloze_benchmark.test.py \
  --data-path benchmark/data/hukukbert_v1_cloze.jsonl \
  --models \
    /workspace/models/hukukbert-base-48k-cased/checkpoint-30500 \
    /workspace/models/hukukbert-base-48k-cased/checkpoint-22000 \
    /workspace/models/hukukbert-base-48k-cased/checkpoint-20000 \
  --topk 3 \
  --device auto \
  --output-json benchmark/results/cloze_eval.json
```

## Checkpoint Results

| Model Checkpoint | N | Top-1 Accuracy | Top-1 95% CI | Top-3 Accuracy | Top-3 95% CI |
|---|---:|---:|---:|---:|---:|
| /workspace/models/hukukbert-base-48k-cased/checkpoint-30500 | 750 | 83.73% | [80.92%, 86.20%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-29500 | 750 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-29000 | 750 | 83.87% | [81.06%, 86.33%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-28500 | 750 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-28000 | 750 | 83.87% | [81.06%, 86.33%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-27500 | 750 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-27000 | 750 | 83.47% | [80.64%, 85.95%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-26500 | 750 | 83.60% | [80.78%, 86.08%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-26000 | 750 | 83.87% | [81.06%, 86.33%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-25500 | 750 | 83.87% | [81.06%, 86.33%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-25000 | 750 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-24500 | 750 | 83.73% | [80.92%, 86.20%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-24000 | 750 | 83.60% | [80.78%, 86.08%] | 98.93% | [97.91%, 99.46%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-23000 | 750 | 83.73% | [80.92%, 86.20%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-22500 | 750 | 83.73% | [80.92%, 86.20%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-22000 | 750 | 84.13% | [81.35%, 86.57%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-21500 | 750 | 84.13% | [81.35%, 86.57%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-21000 | 750 | 83.73% | [80.92%, 86.20%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-20500 | 750 | 84.13% | [81.35%, 86.57%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-20000 | 750 | 84.13% | [81.35%, 86.57%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-18500 | 750 | 83.73% | [80.92%, 86.20%] | 98.53% | [97.39%, 99.18%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-17000 | 750 | 83.33% | [80.50%, 85.83%] | 98.53% | [97.39%, 99.18%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-19000 | 750 | 83.33% | [80.50%, 85.83%] | 98.53% | [97.39%, 99.18%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-19500 | 750 | 83.33% | [80.50%, 85.83%] | 98.93% | [97.91%, 99.46%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-18000 | 750 | 83.07% | [80.22%, 85.58%] | 98.80% | [97.74%, 99.37%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-13000 | 750 | 83.07% | [80.22%, 85.58%] | 98.53% | [97.39%, 99.18%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-10500 | 750 | 82.53% | [79.65%, 85.08%] | 98.40% | [97.22%, 99.08%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-11000 | 750 | 82.13% | [79.23%, 84.71%] | 98.13% | [96.89%, 98.88%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-11500 | 750 | 82.00% | [79.09%, 84.58%] | 98.67% | [97.56%, 99.27%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-9500 | 750 | 81.60% | [78.67%, 84.21%] | 98.40% | [97.22%, 99.08%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-9000 | 750 | 81.47% | [78.53%, 84.08%] | 98.00% | [96.73%, 98.78%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-8500 | 750 | 80.93% | [77.97%, 83.58%] | 98.27% | [97.06%, 98.98%] |
| /workspace/models/hukukbert-base-48k-cased/checkpoint-8000 | 750 | 80.67% | [77.69%, 83.33%] | 98.00% | [96.73%, 98.78%] |
| /workspace/backup/hukukbert-base-48k-cased/checkpoint-6000 | 750 | 80.67% | [77.69%, 83.33%] | 97.87% | [96.56%, 98.68%] |
| /workspace/backup/hukukbert-base-48k-cased/checkpoint-7000 | 750 | 80.67% | [77.69%, 83.33%] | 98.13% | [96.89%, 98.88%] |
| /workspace/backup/hukukbert-base-48k-cased/checkpoint-5000 | 750 | 80.53% | [77.55%, 83.21%] | 97.87% | [96.56%, 98.68%] |
| /workspace/backup/hukukbert-base-48k-cased/checkpoint-4000 | 750 | 79.07% | [76.01%, 81.83%] | 97.73% | [96.40%, 98.58%] |
| dbmdz/bert-base-turkish-128k-cased | 750 | 71.87% | [68.54%, 74.97%] | 95.33% | [93.58%, 96.63%] |
| dbmdz/bert-base-turkish-cased | 750 | 64.53% | [61.04%, 67.88%] | 93.60% | [91.62%, 95.14%] |

## Performance Gain vs Baselines

Best observed HukukBERT scores in this table are **84.13% Top-1** and **98.93% Top-3**.

- vs `dbmdz/bert-base-turkish-128k-cased` (71.87% / 95.33%): **+12.26 points Top-1**, **+3.60 points Top-3**
- vs `dbmdz/bert-base-turkish-cased` (64.53% / 93.60%): **+19.60 points Top-1**, **+5.33 points Top-3**
