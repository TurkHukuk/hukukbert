# Reproducibility: Model Configuration & Training Hyperparameters

This document corresponds to **Appendix F** of the paper and records the complete
model configuration, the exact parameter count, the random seed, and the
domain-adaptive pre-training (DAPT) recipe.

## Model architecture

| Setting | Value |
|---|---|
| Backbone | BERT, encoder-only (`BertForMaskedLM`) |
| Total parameters | **122,961,801** (~123.0M) |
| Vocabulary size | 48,009 (WordPiece) |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Feed-forward (intermediate) size | 3072 |
| Activation | GELU |
| Hidden / attention dropout | 0.1 / 0.1 |
| LayerNorm epsilon | 1e-12 |
| Max position embeddings | 512 |
| Type vocabulary size | 2 |
| Position embedding type | absolute |
| Initializer range | 0.02 |
| Tied input/output embeddings | yes |
| Weights dtype | float32 |
| Transformers version | 4.57.1 |

The canonical configuration is in `config.json`.

### Exact parameter count

For this architecture the total is:

```
params = 769 * vocab_size + 86,042,880
       = 769 * 48,009 + 86,042,880
       = 122,961,801
```

Breakdown:

- **Embeddings:** 37,267,200 — token `48,009 x 768 = 36,870,912`; position `512 x 768`; token-type `2 x 768`; LayerNorm.
- **12 encoder layers:** 85,054,464.
- **MLM head:** 640,137 — the decoder weight is tied to the token embeddings, so only its bias and the transform layer are counted.

Sanity check: the same formula gives `769 * 30,522 + 86,042,880 = 109,514,298`
for a 30,522-vocab `bert-base` — i.e. the familiar ~110M. HukukBERT is larger
only because of the bigger 48,009-token embedding/decoder table.

## Domain-adaptive pre-training (run of record)

| Setting | Value |
|---|---|
| Objective | Masked language modeling (hybrid masking) |
| Optimizer | AdamW |
| Learning-rate schedule | Linear |
| Peak learning rate | 1e-5 |
| Weight decay | 0.01 |
| Epochs | 2 |
| Effective batch size | 960 (base 192 x 5 gradient-accumulation steps) |
| Max sequence length | 512 |
| Random seed | 42 |
| Hardware | 1x NVIDIA H200 SXM |
| Wall-clock | ~19 hours |

### Hybrid masking (overall MLM probability = 0.25)

| Component | Share |
|---|---|
| Whole-word masking | 20% |
| Token span masking | 20% |
| Word span masking | 30% |
| Keyword masking | 30% |
| Keyword lexicon | 40,000+ curated Turkish legal terms |

### Values not fixed in the paper

These knobs were **not** reported in the paper and are therefore not guaranteed.
Set them to your run-of-record values from the training logs (the authors' v2
DAPT launch used the values shown in parentheses):

- **Adam beta2** (v2: `0.98`; default `0.999`), **Adam epsilon** (v2: `1e-6`; default `1e-8`).
- **Warmup steps** (v2: `500`).
- **Token-span mean/max length** (v2: `3` / `8`; default `3` / `10`).
- **Word-span mean/max length** (v2: `2` / `5`; default `3` / `10`).
- **Keyword fallback** (v2: `shared`; default `none`), **keyword match** (v2: `longest`; default `random`).

## Data

- **Pre-training corpus:** ~2.3M documents (19 GB), deduplicated via MinHash LSH
  (num_perm=256, threshold 0.90) and balanced across legal sub-domains.
- **Train/validation/test manifests:** see `data/SPLITS.md`.
- **Legal Cloze Test (intrinsic test set):** 750 items —
  <https://huggingface.co/datasets/turkhukuk/hukukbert-cloze-benchmark>.
