"""
Real-Data Calibration & Validation Benchmark (Amazon ML Challenge 2026)
Follows AGENT.md controlled iteration protocol.
Trains on a stratified sample of real train data, validates on held-out split,
evaluates macro F0.5, inspects false positives/negatives, and calibrates threshold.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import random
import unicodedata
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib

# -------------------------------------------------------------
# 1. Text Normalization & Component Extraction
# -------------------------------------------------------------
LEGAL_SUFFIXES_REGEX = r'\b(corp|corporation|incorporated|inc|ltd|limited|pvt|private|llc|llp|gmbh|ag|sa|sarl|sas|sasu|plc|bv|nv|spa|srl|sl|cie|co|company)\b'

def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = strip_accents(text.lower().strip())
    text = re.sub(r'\b([a-z])\.(?:\s*([a-z])\.?)+', lambda m: m.group(0).replace('.', '').replace(' ', ''), text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_tokens(text: str) -> set:
    if not isinstance(text, str):
        return set()
    cleaned = clean_text(text)
    return {t for t in cleaned.split() if len(t) >= 3 and t not in {'and', 'the', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc'}}

def extract_numbers_and_pins(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d{2,6}\b', text))

def parse_address_simple(addr: str) -> dict:
    if not isinstance(addr, str) or not addr.strip():
        return {'num': '', 'pin': '', 'tokens': set()}
    nums = re.findall(r'\b\d{1,6}(?:[a-zA-Z]|bis|ter)?\b', addr)
    pins = re.findall(r'\b\d{5,6}\b', addr)
    return {
        'num': nums[0].lower() if nums else '',
        'pin': pins[0] if pins else '',
        'tokens': extract_tokens(addr)
    }

# -------------------------------------------------------------
# 2. Pairwise Fast Feature Calculation
# -------------------------------------------------------------
def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else 0.0

def lev_ratio(s1, s2):
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def compute_features(s1_row, c_row):
    n1, n2 = clean_text(s1_row['business_name']), clean_text(c_row['business_name'])
    a1, a2 = clean_text(s1_row['business_address']), clean_text(c_row['business_address'])

    # Name features
    toks1, toks2 = s1_row['name_tokens'], c_row['name_tokens']
    n_jac = jaccard(toks1, toks2)
    n_lev = lev_ratio(n1, n2)
    
    # Acronym match
    acr1 = ''.join(t[0] for t in n1.split() if len(t) > 1)[:5]
    acr2 = ''.join(t[0] for t in n2.split() if len(t) > 1)[:5]
    acr_match = 1.0 if (len(acr1) >= 2 and (acr1 == n2 or acr2 == n1 or acr1 == acr2)) else 0.0

    # Address features
    p1, p2 = s1_row['addr_parsed'], c_row['addr_parsed']
    a_jac = jaccard(p1['tokens'], p2['tokens'])
    a_lev = lev_ratio(a1, a2) if (a1 and a2) else 0.0

    # Number match
    if p1['num'] and p2['num']:
        num_score = 1.0 if p1['num'] == p2['num'] else -1.0
        missing_num = 0.0
    else:
        num_score = 0.0
        missing_num = 1.0

    # PIN match
    if p1['pin'] and p2['pin']:
        pin_score = 1.0 if p1['pin'] == p2['pin'] else -1.0
        missing_pin = 0.0
    else:
        pin_score = 0.0
        missing_pin = 1.0

    has_missing_addr = 1.0 if (not a1 or not a2) else 0.0
    len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)

    return [
        n_jac, n_lev, acr_match,
        a_jac, a_lev, num_score, missing_num,
        pin_score, missing_pin, has_missing_addr, len_diff
    ]

FEATURE_COLS = [
    'name_jaccard', 'name_lev', 'acronym_match',
    'addr_jaccard', 'addr_lev', 'num_score', 'missing_num',
    'pin_score', 'missing_pin', 'has_missing_addr', 'name_len_diff'
]

# -------------------------------------------------------------
# 3. Macro F0.5 Metric
# -------------------------------------------------------------
def compute_macro_f05(ground_truth_map: dict, prediction_map: dict):
    scores = []
    for s1_id, raw_true in ground_truth_map.items():
        true_set = set(raw_true)
        pred_set = set(prediction_map.get(s1_id, []))
        if len(true_set) == 0:
            score = 1.0 if len(pred_set) == 0 else 0.0
            scores.append(score)
        else:
            if len(pred_set) == 0:
                scores.append(0.0)
            else:
                tp = len(true_set & pred_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                if p + r == 0:
                    scores.append(0.0)
                else:
                    scores.append((1.25 * p * r) / (0.25 * p + r))
    return np.mean(scores)

# -------------------------------------------------------------
# 4. Main Validation Workflow
# -------------------------------------------------------------
def main():
    print("=" * 65)
    print("  CALIBRATION BENCHMARK ON REAL COMPETITION DATA  ")
    print("=" * 65)

    N_TRAIN_S1 = 40000
    N_VAL_S1 = 10000
    TOTAL_S1 = N_TRAIN_S1 + N_VAL_S1

    print(f"[1/5] Loading {TOTAL_S1:,} S1 entities and ground truth...")
    s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=TOTAL_S1)
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=TOTAL_S1)
    
    gt_map = {}
    all_true_cands = set()
    for _, r in gt_df.iterrows():
        m = [x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip() and x.strip() != 'nan']
        gt_map[r['source1_entity_id']] = m
        all_true_cands.update(m)

    s1_ids = list(s1_df['entity_id'])
    train_s1_ids = set(s1_ids[:N_TRAIN_S1])
    val_s1_ids = set(s1_ids[N_TRAIN_S1:])

    print(f"  Train S1: {len(train_s1_ids):,} | Val S1: {len(val_s1_ids):,}")
    print(f"  True candidate matches needed: {len(all_true_cands):,}")

    # Load S2 and S3 candidate records
    print("[2/5] Loading relevant S2 and S3 candidate records...")
    cands_dict = {}
    for src in ['train_source2.tsv', 'train_source3.tsv']:
        for chunk in pd.read_csv(f'dataset/train/{src}', sep='\t', chunksize=200000):
            matched = chunk[chunk['entity_id'].isin(all_true_cands)]
            for _, r in matched.iterrows():
                cands_dict[r['entity_id']] = r.to_dict()
            if len(cands_dict) >= len(all_true_cands):
                break

    # Also load a background pool of negative candidates from S2 and S3 (e.g. 50k records)
    bg_s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=30000)
    bg_s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=30000)
    for _, r in pd.concat([bg_s2, bg_s3], ignore_index=True).iterrows():
        if r['entity_id'] not in cands_dict:
            cands_dict[r['entity_id']] = r.to_dict()

    print(f"  Total candidate pool loaded: {len(cands_dict):,} records")

    # Pre-parse tokens & addresses
    print("[3/5] Pre-parsing tokens & building multi-key inverted index...")
    s1_dict = {}
    for _, r in s1_df.iterrows():
        d = r.to_dict()
        d['name_tokens'] = extract_tokens(r['business_name'])
        d['addr_parsed'] = parse_address_simple(r['business_address'])
        d['addr_nums'] = extract_numbers_and_pins(r['business_address'])
        s1_dict[r['entity_id']] = d

    inv_name = {}
    inv_num = {}
    for cid, r in cands_dict.items():
        r['name_tokens'] = extract_tokens(r['business_name'])
        r['addr_parsed'] = parse_address_simple(r['business_address'])
        r['addr_nums'] = extract_numbers_and_pins(r['business_address'])
        country = r['country']
        for tok in r['name_tokens']:
            inv_name.setdefault((country, tok), []).append(cid)
        for num in r['addr_nums']:
            inv_num.setdefault((country, num), []).append(cid)

    # Candidate generation
    print("  Generating candidates for Train & Val...")
    def get_candidates_for_s1(sid):
        r = s1_dict[sid]
        country = r['country']
        cands = set()
        for tok in r['name_tokens']:
            for cid in inv_name.get((country, tok), []):
                cands.add(cid)
        for num in r['addr_nums']:
            for cid in inv_num.get((country, num), []):
                cands.add(cid)
        return list(cands)

    train_pairs = []
    val_pairs = []
    val_cands_dict = {}

    for sid in train_s1_ids:
        # Include all true matches in training + mined negatives
        cands = set(get_candidates_for_s1(sid))
        for mid in gt_map.get(sid, []):
            if mid in cands_dict:
                cands.add(mid)
        # Cap negatives per entity to 30 for training balance
        negatives = [c for c in cands if c not in gt_map.get(sid, [])]
        if len(negatives) > 30:
            negatives = random.sample(negatives, 30)
        final_cands = set(negatives) | (set(gt_map.get(sid, [])) & set(cands_dict.keys()))
        for cid in final_cands:
            train_pairs.append((sid, cid))

    val_true_pairs = 0
    val_captured_pairs = 0
    for sid in val_s1_ids:
        cands = get_candidates_for_s1(sid)
        val_cands_dict[sid] = cands
        true_set = set(gt_map.get(sid, [])) & set(cands_dict.keys())
        val_true_pairs += len(true_set)
        val_captured_pairs += len(true_set & set(cands))
        for cid in cands:
            val_pairs.append((sid, cid))

    print(f"  Train Pairs: {len(train_pairs):,} | Val Pairs: {len(val_pairs):,}")
    if val_true_pairs > 0:
        print(f"  Val Blocking Recall: {val_captured_pairs}/{val_true_pairs} ({val_captured_pairs/val_true_pairs*100:.2f}%)")

    # Feature extraction
    print("[4/5] Extracting pairwise features for training...")
    X_train, y_train = [], []
    for sid, cid in train_pairs:
        feat = compute_features(s1_dict[sid], cands_dict[cid])
        X_train.append(feat)
        y_train.append(1 if cid in gt_map.get(sid, []) else 0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"  X_train shape: {X_train.shape} | Positive pairs: {np.sum(y_train):,} ({np.mean(y_train)*100:.1f}%)")

    # Train LightGBM model
    print("  Training LightGBM Classifier...")
    clf = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)

    # Feature importance
    print("\n  --- Top Feature Importances ---")
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for col, v in imp.items():
        print(f"    {col:20s}: {v}")

    # Validation evaluation & threshold tuning
    print("\n[5/5] Scoring Validation Set & Sweeping Decision Threshold...")
    X_val = []
    for sid, cid in val_pairs:
        X_val.append(compute_features(s1_dict[sid], cands_dict[cid]))
    X_val = np.array(X_val, dtype=np.float32) if val_pairs else np.empty((0, len(FEATURE_COLS)))

    val_probs = clf.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])

    val_gt = {sid: gt_map.get(sid, []) for sid in val_s1_ids}

    # Bipartite collective resolution function
    def run_collective_eval(primary_t, secondary_t=None):
        if secondary_t is None:
            secondary_t = primary_t + 0.15
        s1_cands = {sid: [] for sid in val_s1_ids}
        cand_claims = {}
        for (sid, cid), p in zip(val_pairs, val_probs):
            if p >= primary_t:
                s1_cands[sid].append((cid, p))
                cand_claims.setdefault(cid, []).append((p, sid))

        cand_winner = {}
        for cid, claims in cand_claims.items():
            claims.sort(reverse=True, key=lambda x: x[0])
            cand_winner[cid] = claims[0][1]

        preds = {sid: [] for sid in val_s1_ids}
        for sid, c_list in s1_cands.items():
            if not c_list:
                continue
            c_list.sort(reverse=True, key=lambda x: x[1])
            top_cid, top_p = c_list[0]
            if cand_winner.get(top_cid) == sid:
                preds[sid].append(top_cid)
            for cid, p in c_list[1:]:
                if cand_winner.get(cid) == sid and p >= secondary_t and (top_p - p) <= 0.25:
                    preds[sid].append(cid)
        return preds

    best_t = 0.50
    best_f05 = 0.0
    for t in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = run_collective_eval(primary_t=t)
        f05 = compute_macro_f05(val_gt, preds)
        print(f"  Threshold {t:.2f} -> Macro F0.5: {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_t = t

    print("\n" + "=" * 65)
    print(f"  VALIDATION RESULTS: Optimal Threshold = {best_t:.2f} | Macro F0.5 = {best_f05:.4f}")
    print("=" * 65)

    # False Positive & False Negative Analysis (AGENT.md Requirement)
    best_preds = run_collective_eval(primary_t=best_t)
    fps = []
    fns = []
    for sid in val_s1_ids:
        true_set = set(val_gt[sid])
        pred_set = set(best_preds.get(sid, []))
        for cid in (pred_set - true_set):
            fps.append((sid, cid))
        for cid in (true_set - pred_set):
            fns.append((sid, cid))

    print(f"\n  Validation Error Summary:")
    print(f"    Total False Positives (wrong merges): {len(fps)}")
    print(f"    Total False Negatives (missed links): {len(fns)}")

    if fps:
        print("\n  --- Sample False Positives (Audited per AGENT.md) ---")
        for sid, cid in fps[:5]:
            r1, r2 = s1_dict[sid], cands_dict.get(cid, {})
            print(f"    • S1:  {r1.get('business_name')} @ {r1.get('business_address')}")
            print(f"      FP:  {r2.get('business_name')} @ {r2.get('business_address')}")

    if fns:
        print("\n  --- Sample False Negatives (Audited per AGENT.md) ---")
        for sid, cid in fns[:5]:
            r1, r2 = s1_dict[sid], cands_dict.get(cid, {})
            print(f"    • S1:  {r1.get('business_name')} @ {r1.get('business_address')}")
            print(f"      FN:  {r2.get('business_name')} @ {r2.get('business_address')}")

if __name__ == '__main__':
    main()
