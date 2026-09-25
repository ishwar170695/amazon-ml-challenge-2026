# Amazon ML Challenge 2026 - Business Entity Resolution

End-to-end entity resolution pipeline linking deduplicated reference businesses (Source 1) with noisy multi-source records (Source 2 & Source 3).

## Setup

```bash
pip install numpy pandas scikit-learn lightgbm recordlinkage
```

## Running the Pipeline

Run benchmark / synthetic evaluation:
```bash
python business_entity_resolution.py
```

Run on competition dataset (place files in `dataset/train/` and `dataset/test/`):
```bash
python business_entity_resolution.py --real
```

Output matches are generated in `output/matching_results.tsv`.
