# HukukBERT

A domain-specific BERT model for Turkish legal text, pretrained from scratch on 6 million unique court decisions with a custom 48K WordPiece tokenizer.

**HukukBERT is developed by [TurkHukuk.ai](https://www.turkhukuk.ai)**

## Key Results

| Model | Top-1 Accuracy | Top-1 95% CI | Top-3 Accuracy | Top-3 95% CI |
|---|---:|---:|---:|---:|
| turkhukuk.ai/hukukbert | 83,87% | [81,06% – 86,33%] | 98,80% | [97,74% – 99,37%] |
| KocLab-Bilkent/BERTurk-Legal | 75,07% | [71,85% – 78,03%] | 96,00% | [94,35% – 97,18%] |
| dbmdz/bert-base-turkish-128k-cased | 71,87% | [68,54% – 74,97%] | 95,33% | [93,58% – 96,63%] |
| dbmdz/bert-base-turkish-cased | 64,53% | [61,04% – 67,88%] | 93,60% | [91,62% – 95,14%] |
| newmindai/Mursit-Large | 62,53% | [59,01% – 65,93%] | 94,40% | [92,52% – 95,83%] |
| ytu-ce-cosmos/turkish-large-bert-cased | 61,20% | [57,66% – 64,62%] | 91,07% | [88,81% – 92,90%] |
| boun-tabilab/TabiBERT | 48,40% | [44,84% – 51,98%] | 88,40% | [85,91% – 90,50%] |

`turkhukuk.ai/hukukbert` outperforms the strongest baseline in this table (`KocLab-Bilkent/BERTurk-Legal`) by **+8,80 points** on Top-1 and **+2,80 points** on Top-3 accuracy on the legal cloze benchmark (n=750).

## Model Details

| | |
|---|---|
| **Architecture** | BERT-base (12 layers, 768 hidden, 12 heads) |
| **Tokenizer** | Custom 48K WordPiece, trained on Turkish legal corpus |
| **Pretraining corpus** | ~6M unique court decisions, mevzuat text, and various legal articles (Yargıtay, İstinaf, İlk Derece, Danıştay, AYM, Mevzuat, legal articles) |
| **Deduplication** | MinHash + LSH on 11M original decisions → 6M unique |
| **Casing** | Cased |

## Why a Domain-Specific Tokenizer?

General Turkish tokenizers fragment legal terminology into meaningless subwords. For example, *"temerrüt"* may be split into `["te", "##mer", "##rüt"]` by a general tokenizer. HukukBERT's 48K legal-domain tokenizer recognizes such terms as single tokens, preserving semantic meaning for downstream tasks.

## Repository Contents

```
hukukbert/
├── README.md                ← this file
├── LICENSE                  ← Apache 2.0 (code)
├── LICENSE-DATA             ← CC BY 4.0 (benchmark data)
├── CITATION.cff             ← citation metadata
└── benchmark/
    ├── README.md            ← benchmark usage & detailed results
    ├── data/
    │   └── hukukbert_v1_cloze.jsonl  (750 cloze items)
    └── scripts/
        └── cloze_benchmark_test.py   (evaluation script)
```

### What's Included

- ✅ Cloze benchmark dataset (750 items, Turkish legal domain)
- ✅ Evaluation script with confidence intervals
- ✅ Full checkpoint results across training progression

### What's Not Included

- ❌ Model weights (available for research collaboration — see below)
- ❌ Tokenizer files
- ❌ Training data or training pipeline

## Benchmark

The benchmark is a cloze test for Turkish legal language modeling. Each item contains a legal sentence with a single `[MASK]` token, a set of candidate options, and one gold answer. Results are reported with Top-1 and Top-3 accuracy plus Wilson 95% confidence intervals.

See [benchmark/README.md](benchmark/README.md) for usage instructions and [full checkpoint results](benchmark/README.md#checkpoint-results).

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
