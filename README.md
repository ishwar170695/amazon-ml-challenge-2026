# Amazon ML Challenge 2026 — Business Entity Resolution

Entity resolution pipeline linking deduplicated reference businesses (**Source 1**) with noisy multi-source records (**Source 2** & **Source 3**) across multiple countries (United States, India, France).

---

## 1. Performance & Best Validated Results

Evaluated on a strict, leak-free **3-way entity-level split** (17.5k Train / 2.5k Validation / 5.0k Frozen Held-Out Test):

| Metric | Frozen Test Set | Validation Set (Uncapped Threshold) |
| :--- | :---: | :---: |
| **Macro $F_{0.5}$** | **0.9773** | **0.9811** |
| **United States $F_{0.5}$** | **0.9834** | **0.9870** |
| **India $F_{0.5}$** | **0.9688** | **0.9722** |
| **Blocking Recall** | **99.62%** (India: 99.26%) | **99.62%** |
| **Open-Set France Stress Test** | **100.0%** (8/8 matches) | **100.0%** |

---

## 2. End-to-End Pipeline Architecture

```
Raw Sources (S1, S2, S3)
  │
  ▼
[1. Multi-Script Preprocessing & Normalization]
  ├── Indic script transliteration via anyascii (Devanagari, Tamil, Telugu → Latin)
  ├── Indic transliterated legal suffixes ('praivet', 'limirrd', 'elelpi', 'pra li')
  ├── Soft-C phonetic representation: c(?=[eiy]) → s, c → k
  ├── Compact brand extraction & honorific stripping (M/s, Shri, Dr, .com)
  └── French legal suffixes (EURL, SCI, SASU, SARL, etc.) & accent normalization
  │
  ▼
[2. High-Recall Multi-Key Inverted Index Blocking] (~99.62% recall)
  ├── Significant name tokens + 2-6 digit address numbers + house codes
  ├── Phonetic skeleton tokens (PH) + Unit keys (UK) + Compact brand tokens (CB)
  └── Distinctive address tokens (recovering rebranded & phonetic drift entities)
  │
  ▼
[3. 21 Fast Decomposed Pairwise Features] (~0.6 µs per pair)
  ├── RapidFuzz C++ token sort ratio, Levenshtein ratio, char 3-gram Jaccard
  ├── Precomputed sparse CSR TF-IDF cosines (word (1,2)-grams & char (3,5)-grams)
  ├── Phonetic skeleton similarity (phs), compact brand similarity (cbs)
  ├── Unit key match (ukm), street/address Levenshtein (slv, alv)
  └── Number & postcode match / mismatch / missingness indicators
  │
  ▼
[4. LightGBM Matching Classifier]
  └── 300 trees, 63 leaves, max depth 8, balanced class weights
  │
  ▼
[5. Competitive Collective Bipartite Resolution]
  ├── 1-to-1 primary match locking with country-calibrated thresholds
  ├── Quality-controlled secondary multi-matches (max 12 matches per S1)
  └── Automatic singleton prediction if no candidate exceeds threshold
  │
  ▼
Submission Output: output/matching_results.tsv & output/candidate_pairs.tsv
```

---

## 3. Quick Start & Setup

### Environment Requirements
```bash
pip install numpy pandas scipy scikit-learn lightgbm rapidfuzz anyascii
```

### Reproducing the Pipeline

#### A. Train the Model
Trains the LightGBM classifier on the 17.5k entity split and saves artifacts to `artifacts/model_v3.pkl`:
```bash
python run_pipeline.py --train
```

#### B. Run Leak-Free Validation
Runs entity-level 3-way validation, evaluates threshold sweeps, and executes the French open-set synthetic stress test:
```bash
python run_pipeline.py --validate
```

#### C. Generate Final Submission Files
Generates the official submission files streamed country-by-country (memory bounded < 2.5 GB) and automatically runs official format validation:
```bash
python run_pipeline.py --predict
```

*For quick local testing on a subset:*
```bash
# Test first 1,000 entities in France
python run_pipeline.py --predict --country France --limit 1000

# Test first 5,000 entities in India
python run_pipeline.py --predict --country India --limit 5000
```

#### Output Files
* `output/matching_results.tsv`: TSV with columns `source1_id`, `matched_ids` (comma-separated, or empty for singletons).
* `output/candidate_pairs.tsv`: TSV with columns `source1_id`, `candidate_ids` (all candidate pairs considered).

Both outputs are automatically validated against `6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py`.

---

## 4. Key Experiments & Engineering Ledger

A summary of verified hypotheses and architectural iterations (details in [`AGENT.md`](file:///c:/Users/ishu/Downloads/ml_hack/AGENT.md) and [`experiments/ledger.jsonl`](file:///c:/Users/ishu/Downloads/ml_hack/experiments/ledger.jsonl)):

1. **Indic Transliteration (`iter_008`)**:
   - *Problem*: Tamil/Devanagari records in S2/S3 had 0 token overlap with English S1 records.
   - *Solution*: Universal `anyascii` transliteration before normalization.
   - *Impact*: Jumped macro $F_{0.5}$ from 0.830 to 0.9509.
2. **Distinctive Address Blocking (`iter_012`)**:
   - *Problem*: Rebranded entities and phonetic drift missed traditional name blockers.
   - *Solution*: Distinctive address token indexing (excluding stopwords) with a 300-key frequency cap.
   - *Impact*: India blocking recall increased from 92.3% to **99.26%**; total blocking recall reached **99.62%**.
3. **Indic Transliterated Legal Suffixes (`iter_020`)**:
   - *Problem*: Transliterated legal terms (`praivet`, `limirrd`) were not stripped, diluting name tokens.
   - *Solution*: Expanded regex to normalize non-Latin legal variants.
   - *Impact*: India test $F_{0.5}$ improved from 0.9618 to **0.9688** (+0.70 pp).
4. **Guard Override Elimination (`iter_020`)**:
   - *Problem*: Forensic audit revealed 87.7% of false negatives had model confidence $P \ge 0.88$ (many $P > 0.999$) but were suppressed by downstream string-similarity guards.
   - *Solution*: Removed rigid guards in favor of pure bipartite collective winner assignment with calibrated thresholds.
   - *Impact*: Test macro $F_{0.5}$ jumped from 0.9599 to **0.9773**.
5. **Soft-C Phonetic Normalization (`iter_021`)**:
   - *Problem*: Collisions and mismatches in words like `finance` vs `phainems` (soft-c vs hard-k).
   - *Solution*: Context-aware phonetic rule `c(?=[eiy]) → s`, else `c → k`.
   - *Impact*: Val macro +0.12 pp, India Val +0.17 pp.
6. **Threshold Ceiling Uncapping**:
   - *Problem*: Grid search previously capped at `0.98` masked the true calibration optimum of the LightGBM probability distribution.
   - *Solution*: Expanded threshold grid up to `0.995`.
   - *Impact*: Validation macro $F_{0.5}$ reached **0.9811** (US: 0.9870, India: 0.9722).
