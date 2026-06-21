# Data Splits

HukukBERT is domain-adaptively pre-trained on a balanced Turkish legal corpus and
evaluated on two tasks. This document specifies the splits referenced in the
paper's Methodology and in response to Reviewers C1 / A3.

## Pre-training corpus

- **Source:** Turkish legal corpus (case law, legislation, scholarship),
  deduplicated with MinHash LSH (`num_perm=256`, similarity threshold `0.90`)
  and balanced across sub-domains.
- **Size:** ~19 GB / ~2.3M documents.
- **Format:** a `datasets.save_to_disk` directory with `train` and `validation`
  splits, tokenized at sequence length 512 with stride 0.

### Train / validation / test splits

Consistent with the paper (Appendix F), the exact train/validation/test split
sizes and manifests are **available from the authors on reasonable request** and
are not enumerated here. For context, the pre-training corpus totals ~2.3M
documents (≈19 GB), tokenized at sequence length 512 for masked language
modeling.

## Intrinsic evaluation — Legal Cloze Test

- Test-only benchmark, **N = 750**, no training split.
- Synthetically generated with LLMs and independently verified by a qualified
  lawyer and a software engineer; contamination-free by construction (the
  passages are absent from the pre-training corpus).
- Public: <https://huggingface.co/datasets/turkhukuk/hukukbert-cloze-benchmark>

## Downstream evaluation — court-decision segmentation

- Fine-tuning protocol (identical across all compared models): sliding window
  512 / stride 256, learning rate 3e-5, effective batch size 16, 4 epochs,
  B-tag weight 5.0, automatic class weights.
- If releasing this task, add its train/validation/test split manifests here.
