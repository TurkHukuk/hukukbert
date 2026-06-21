# HukukBERT

A domain-specific BERT model for Turkish legal text, built by domain-adaptive pre-training (DAPT) on a balanced ~2.3M-document (19 GB) Turkish legal corpus with a custom 48K WordPiece tokenizer.

**HukukBERT is developed by [TurkHukuk.ai](https://www.turkhukuk.ai)**

## Key Results

| Model | Top-1 Accuracy | Top-1 95% CI | Top-3 Accuracy | Top-3 95% CI |
|---|---:|---:|---:|---:|
| turkhukuk.ai/hukukbert | 84.40% | [81.63%, 86.82%] | 98.80% | [97.74%, 99.37%] |
| newmindai/Mursit-Large | 78.80% | [75.73%, 81.57%] | 97.73% | [96.40%, 98.58%] |
| KocLab-Bilkent/BERTurk-Legal | 75.47% | [72.26%, 78.41%] | 96.00% | [94.35%, 97.18%] |
| dbmdz/bert-base-turkish-128k-cased | 71.87% | [68.54%, 74.97%] | 95.20% | [93.43%, 96.51%] |
| boun-tabilab/TabiBERT | 68.13% | [64.71%, 71.37%] | 95.33% | [93.58%, 96.63%] |
| dbmdz/bert-base-turkish-cased | 63.73% | [60.23%, 67.10%] | 93.47% | [91.47%, 95.02%] |
| ytu-ce-cosmos/turkish-large-bert-cased | 61.60% | [58.07%, 65.01%] | 91.20% | [88.96%, 93.02%] |

`turkhukuk.ai/hukukbert` outperforms the strongest baseline in this table (`newmindai/Mursit-Large`) by **+5,60 points** on Top-1 and **+1,07 points** on Top-3 accuracy on the legal cloze benchmark (n=750).

## Model Details

| | |
|---|---|
| **Architecture** | BERT-base (12 layers, 768 hidden, 12 heads) |
| **Tokenizer** | Custom 48K WordPiece, trained on Turkish legal corpus |
| **Pretraining corpus** | ~2.3M documents (19 GB), balanced across court decisions, mevzuat, and legal scholarship (Yargıtay, İstinaf, İlk Derece, Danıştay, AYM, Mevzuat, legal articles) |
| **Deduplication & balancing** | MinHash LSH (num_perm=256, threshold 0.90): ~11M raw docs (27 GB) → ~8.9M (24.6 GB), then sub-domain balancing → ~2.3M (19 GB) |
| **Casing** | Cased |
| **Pretraining method** | Domain-adaptive pre-training (DAPT) — see [REPRODUCIBILITY.md](./REPRODUCIBILITY.md) |

## Why a Domain-Specific Tokenizer?

General Turkish tokenizers fragment legal terminology into meaningless subwords. For example, *"temerrüt"* may be split into `["te", "##mer", "##rüt"]` by a general tokenizer. HukukBERT's 48K legal-domain tokenizer recognizes such terms as single tokens, preserving semantic meaning for downstream tasks.

## Repository Contents

```
hukukbert/
├── README.md                ← this file
├── REPRODUCIBILITY.md       ← model config, parameter count, seed, DAPT recipe (Appendix F)
├── config.json              ← exact Hugging Face model configuration
├── LICENSE                  ← Apache 2.0 (code)
├── LICENSE-DATA             ← CC BY 4.0 (benchmark data)
├── CITATION.cff             ← citation metadata
├── benchmark/
│   ├── README.md            ← benchmark usage & detailed results
│   ├── data/
│   │   └── hukukbert_v1_cloze.jsonl  (750 cloze items)
│   └── scripts/
│       └── cloze_benchmark_test.py   (evaluation script)
└── data/
    └── SPLITS.md            ← data split spec & manifest template
```

### What's Included

- ✅ Cloze benchmark dataset (750 items, Turkish legal domain)
- ✅ Evaluation script with confidence intervals
- ✅ Full checkpoint results across training progression
- ✅ Reproducibility docs: exact model config, parameter count, seed, the DAPT recipe, and the data-split spec (see [Reproducibility](#reproducibility))

### What's Not Included

- ❌ Model weights (available for research collaboration — see below)
- ❌ Tokenizer files
- ❌ Training data / tokenized splits
- ❌ Training code / pipeline — the configuration and DAPT recipe are documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), but the training scripts, helper modules, and keyword lexicon are not released

## Benchmark

The benchmark is a cloze test for Turkish legal language modeling. Each item contains a legal sentence with a single `[MASK]` token, a set of candidate options, and one gold answer. Results are reported with Top-1 and Top-3 accuracy plus Wilson 95% confidence intervals.

See [benchmark/README.md](benchmark/README.md) for usage instructions and [full checkpoint results](benchmark/README.md#checkpoint-results).

## Reproducibility

Documentation of the exact model configuration and the domain-adaptive
pre-training "run of record", published in support of the paper's Data
Availability Statement (Appendix F). This is **documentation only** — model
weights, training code, and the training corpus are **not** released.

| File | Purpose |
|---|---|
| [`config.json`](config.json) | Exact Hugging Face model configuration (architecture). |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Architecture, exact parameter count (122,961,801), random seed, and the DAPT / masking recipe. |
| [`data/SPLITS.md`](data/SPLITS.md) | Train / validation / test split specification and manifest template. |

## Downstream Applications

HukukBERT serves as the foundation for several downstream Turkish legal NLP tasks:

- **Court decision segmentation** — classifying sections (iddia, savunma, gerekçe, hüküm)
- **Party identification** — detecting and classifying parties (kamu, tüzel, gerçek kişi)
- **Judgment extraction** — extracting structured hüküm from decision text

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{hukukbert2026,
  title     = {HukukBERT: A Domain-Specific Language Model for Turkish Legal Text},
  author    = {Turkoglu, Tansu},
  email     = {tansu@turkhukuk.ai},
  year      = {2026},
  url       = {https://github.com/TurkHukuk/hukukbert},
  publisher = {TurkHukuk.ai}
}
```

## License

- Code: [Apache License 2.0](./LICENSE)
- Benchmark data: [Creative Commons Attribution 4.0](./LICENSE-DATA)
