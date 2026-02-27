# HukukBERT Cloze Benchmark

Detailed usage instructions and full results for the HukukBERT cloze benchmark.
For a high-level overview, see the root [README.md](../README.md).

All paths below are repository-root relative.

## Files

- Dataset: `benchmark/data/hukukbert_v1_cloze.jsonl`
- Evaluation script: `benchmark/scripts/cloze_benchmark_test.py`

## Data Format

Each item is a cloze test: a legal sentence with exactly one `[MASK]` token, a set of candidate options, and one gold answer.

```json
{
  "id": "bert_cloze_0001",
  "sentence": "... [MASK] ...",
  "options": ["A", "B", "C", "D"],
  "gold": "C",
  "metadata": {
    "law_area": "ceza_muhakemesi_hukuku",
    "difficulty": "medium"
  }
}
```

Fields:

| Field | Required | Description |
|---|---|---|
| `id` | ✅ | Unique item identifier |
| `sentence` | ✅ | Legal sentence with exactly one `[MASK]` |
| `options` | ✅ | Candidate answers (list of strings) |
| `gold` | ✅ | Correct answer (must be in `options`) |
| `metadata` | — | Optional: `law_area`, `difficulty`, etc. |

The dataset ships in **JSONL** format (one JSON object per line).

## Running the Benchmark

**Single model:**

```bash
python benchmark/scripts/cloze_benchmark_test.py \
  --data-path benchmark/data/hukukbert_v1_cloze.jsonl \
  --models /path/to/your/model \
  --topk 3 \
  --device auto
```

**Multiple models (comparison run):**

```bash
python benchmark/scripts/cloze_benchmark_test.py \
  --data-path benchmark/data/hukukbert_v1_cloze.jsonl \
  --models \
    /path/to/model-a \
    /path/to/model-b \
    dbmdz/bert-base-turkish-128k-cased \
  --topk 3 \
  --device auto \
  --output-json benchmark/results/cloze_eval.json
```

**Evaluate a HuggingFace Hub model:**

```bash
python benchmark/scripts/cloze_benchmark_test.py \
  --data-path benchmark/data/hukukbert_v1_cloze.jsonl \
  --models dbmdz/bert-base-turkish-128k-cased \
  --topk 3
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--models` | *(required)* | One or more model paths or HuggingFace Hub names |
| `--data-path` | *(required)* | Path to JSONL benchmark file |
| `--topk` | `1` | Report Top-k accuracy (e.g., `3` for Top-1 and Top-3) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--multiword-scoring` | `independent` | `independent` or `pll` (pseudo-log-likelihood) |
| `--max-seq-length` | `512` | Maximum token length |
| `--eval-batch-size` | `32` | Batch size for independent scoring |
| `--pll-variant-batch-size` | `256` | Batch size for PLL variant forwards |
| `--output-json` | — | Save detailed results to JSON file |
| `--verbose` | `false` | Per-item prediction logs |

## Notes

- The script normalizes `[mask]` → `[MASK]` and validates each item before evaluation.
- Accuracy is reported with Wilson 95% confidence intervals.
- Multi-token options (e.g., compound legal terms) can be scored with `--multiword-scoring pll` for more accurate results at the cost of additional forward passes.

## Checkpoint Results

Full training progression of HukukBERT on this benchmark (n=750).

| Checkpoint | Top-1 | Top-1 95% CI | Top-3 | Top-3 95% CI |
|---:|---:|---:|---:|---:|
| 22000 | **84.13%** | [81.35%, 86.57%] | 98.80% | [97.74%, 99.37%] |
| 21500 | 84.13% | [81.35%, 86.57%] | 98.80% | [97.74%, 99.37%] |
| 20500 | 84.13% | [81.35%, 86.57%] | 98.67% | [97.56%, 99.27%] |
| 20000 | 84.13% | [81.35%, 86.57%] | 98.67% | [97.56%, 99.27%] |
| 29000 | 83.87% | [81.06%, 86.33%] | 98.67% | [97.56%, 99.27%] |
| 28000 | 83.87% | [81.06%, 86.33%] | 98.80% | [97.74%, 99.37%] |
| 26000 | 83.87% | [81.06%, 86.33%] | 98.80% | [97.74%, 99.37%] |
| 25500 | 83.87% | [81.06%, 86.33%] | 98.67% | [97.56%, 99.27%] |
| 30500 | 83.73% | [80.92%, 86.20%] | 98.67% | [97.56%, 99.27%] |
| 24500 | 83.73% | [80.92%, 86.20%] | 98.80% | [97.74%, 99.37%] |
| 23000 | 83.73% | [80.92%, 86.20%] | 98.67% | [97.56%, 99.27%] |
| 22500 | 83.73% | [80.92%, 86.20%] | 98.67% | [97.56%, 99.27%] |
| 21000 | 83.73% | [80.92%, 86.20%] | 98.80% | [97.74%, 99.37%] |
| 18500 | 83.73% | [80.92%, 86.20%] | 98.53% | [97.39%, 99.18%] |
| 29500 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| 28500 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| 27500 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| 26500 | 83.60% | [80.78%, 86.08%] | 98.80% | [97.74%, 99.37%] |
| 25000 | 83.60% | [80.78%, 86.08%] | 98.67% | [97.56%, 99.27%] |
| 24000 | 83.60% | [80.78%, 86.08%] | **98.93%** | [97.91%, 99.46%] |
| 27000 | 83.47% | [80.64%, 85.95%] | 98.80% | [97.74%, 99.37%] |
| 17000 | 83.33% | [80.50%, 85.83%] | 98.53% | [97.39%, 99.18%] |
| 19000 | 83.33% | [80.50%, 85.83%] | 98.53% | [97.39%, 99.18%] |
| 19500 | 83.33% | [80.50%, 85.83%] | 98.93% | [97.91%, 99.46%] |
| 18000 | 83.07% | [80.22%, 85.58%] | 98.80% | [97.74%, 99.37%] |
| 13000 | 83.07% | [80.22%, 85.58%] | 98.53% | [97.39%, 99.18%] |
| 10500 | 82.53% | [79.65%, 85.08%] | 98.40% | [97.22%, 99.08%] |
| 11000 | 82.13% | [79.23%, 84.71%] | 98.13% | [96.89%, 98.88%] |
| 11500 | 82.00% | [79.09%, 84.58%] | 98.67% | [97.56%, 99.27%] |
| 9500 | 81.60% | [78.67%, 84.21%] | 98.40% | [97.22%, 99.08%] |
| 9000 | 81.47% | [78.53%, 84.08%] | 98.00% | [96.73%, 98.78%] |
| 8500 | 80.93% | [77.97%, 83.58%] | 98.27% | [97.06%, 98.98%] |
| 8000 | 80.67% | [77.69%, 83.33%] | 98.00% | [96.73%, 98.78%] |
| 7000 | 80.67% | [77.69%, 83.33%] | 98.13% | [96.89%, 98.88%] |
| 6000 | 80.67% | [77.69%, 83.33%] | 97.87% | [96.56%, 98.68%] |
| 5000 | 80.53% | [77.55%, 83.21%] | 97.87% | [96.56%, 98.68%] |
| 4000 | 79.07% | [76.01%, 81.83%] | 97.73% | [96.40%, 98.58%] |

### Baselines

| Model | Top-1 | Top-1 95% CI | Top-3 | Top-3 95% CI |
|---|---:|---:|---:|---:|
| dbmdz/bert-base-turkish-128k-cased | 71.87% | [68.54%, 74.97%] | 95.33% | [93.58%, 96.63%] |
| dbmdz/bert-base-turkish-cased | 64.53% | [61.04%, 67.88%] | 93.60% | [91.62%, 95.14%] |

### Observations

- Top-1 accuracy plateaus around checkpoint 20000–22000 (~84.1%), with no meaningful improvement through checkpoint 30500. This suggests the model has saturated on this benchmark by ~20K steps.
- Top-3 accuracy is remarkably stable across all checkpoints (97.7%–98.9%), indicating the correct answer is almost always in the model's top predictions even early in training.
- Best Top-1 (84.13%) and best Top-3 (98.93%) come from different checkpoints (22000 vs 24000), though the differences are within confidence intervals.
