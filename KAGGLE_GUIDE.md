# 🚀 Kaggle Turnkey Drag-and-Drop Guide (Amazon ML Challenge 2026)

This bundle executes the complete, frozen **0.9773 Test Macro $F_{0.5}$** entity resolution pipeline across all **1,732,544 test S1 entities** in **~15–20 minutes** on Kaggle.

---

## 📦 What's in this Bundle
| File | Purpose |
|---|---|
| `kaggle_pipeline.py` | Standalone, high-throughput parallel predictor (4 vCPUs, Linux CoW, on-the-fly 3-gram feature extraction). |
| `run_kaggle.ipynb` | 3-cell Jupyter Notebook ready to upload directly into Kaggle. |
| `model_v3.pkl` | Frozen production LightGBM model and TF-IDF vectorizers (5.96 MB). |
| `validate_submission.py` | Official submission format validator ensuring 100% compliance. |

---

## ⚡ 3-Step Execution on Kaggle

### Step 1: Create a Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Click **File > Upload Notebook** and select [`run_kaggle.ipynb`](./run_kaggle.ipynb).
3. In the right-hand panel:
   - **Accelerator**: Select **"None"** (gives 4 CPU cores & 30 GB RAM).
   - **Internet**: Toggle to **"On"** (to allow `pip install rapidfuzz anyascii`).

### Step 2: Add Files & Test Data
You need 2 things attached to the notebook:
1. **The Model & Script**:
   - Drag & drop `kaggle_pipeline.py` and `model_v3.pkl` directly into the Kaggle file explorer or upload them as a private dataset.
2. **The Test Dataset**:
   - In the right-hand sidebar, click **+ Add Input**.
   - If you have uploaded your test dataset (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`), select it.
   - *Note: `kaggle_pipeline.py` automatically searches `/kaggle/input/**` recursively. Wherever your test files are, it will locate them instantly.*

### Step 3: Click "Run All"
- Press **Run All** (or run Cell 1, Cell 2, Cell 3).
- **Execution breakdown:**
  - **France (259k entities):** ~2.5 minutes
  - **India (603k entities):** ~4.5 minutes
  - **US (870k entities):** ~8.0 minutes
  - **Total Runtime:** **~15–18 minutes!**

---

## 📥 Download Your Submission
Once complete, Cell 3 provides a direct download link.
Alternatively, in the right sidebar under **Data > Output > /kaggle/working/output**:
- Download **`submission_archive.zip`** (contains both `matching_results.tsv` and `candidate_pairs.tsv`).
- This archive is pre-validated against the official format and ready to upload to the challenge submission portal.
