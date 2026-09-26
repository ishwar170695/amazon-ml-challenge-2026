"""
Fast Parallel Model Training & Threshold Calibration on Real Competition Data
Utilizes all 12 CPU cores for maximum throughput.
Trains LightGBM classifier on real US & India training pairs with hard negative mining.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import time
import random
import pickle
import unicodedata
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import difflib

# -------------------------------------------------------------
# 1. High-Performance Text Normalization & Feature Extraction
# -------------------------------------------------------------
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

def get_tokens(text: str) -> set:
    if not isinstance(text, str):
        return set()
    cleaned = clean_text(text)
    return {t for t in cleaned.split() if len(t) >= 3 and t not in {'and', 'the', 'for', 'ltd', 'pvt', 'inc', 'corp', 'llc'}}

def get_distinctive_numbers(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d{3,6}\b', text))

def parse_address_fast(addr: str) -> dict:
    if not isinstance(addr, str) or not addr.strip():
        return {'num': '', 'pin': '', 'tokens': set()}
    nums = re.findall(r'\b\d{1,6}(?:[a-zA-Z]|bis|ter)?\b', addr)
    pins = re.findall(r'\b\d{5,6}\b', addr)
    return {
        'num': nums[0].lower() if nums else '',
        'pin': pins[0] if pins else '',
        'tokens': get_tokens(addr)
    }

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

def compute_pairwise_features(s1_dict, c_dict):
    n1, n2 = clean_text(s1_dict['business_name']), clean_text(c_dict['business_name'])
    a1, a2 = clean_text(s1_dict['business_address']), clean_text(c_dict['business_address'])

    # Name features
    toks1, toks2 = s1_dict['name_tokens'], c_dict['name_tokens']
    n_jac = jaccard(toks1, toks2)
    n_lev = lev_ratio(n1, n2)

    # Acronym match
    acr1 = ''.join(t[0] for t in n1.split() if len(t) > 1)[:5]
    acr2 = ''.join(t[0] for t in n2.split() if len(t) > 1)[:5]
    acr_match = 1.0 if (len(acr1) >= 2 and (acr1 == n2 or acr2 == n1 or acr1 == acr2)) else 0.0

    # Address features
    p1, p2 = s1_dict['addr_parsed'], c_dict['addr_parsed']
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
# 2. Macro F0.5 Metric
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
    return float(np.mean(scores))

# -------------------------------------------------------------
# 3. Model Training Pipeline
# -------------------------------------------------------------
def train_and_validate():
    print("=" * 65, flush=True)
    print("  TRAINING HIGH-PERFORMANCE LIGHTGBM MODEL ON REAL COMPETITION DATA  ", flush=True)
    print(f"  Available CPU Cores: {os.cpu_count()}", flush=True)
    print("=" * 65, flush=True)

    N_TRAIN_S1 = 20000
    N_VAL_S1 = 5000
    TOTAL_S1 = N_TRAIN_S1 + N_VAL_S1

    t0 = time.time()
    print(f"[1/5] Fast-reading {TOTAL_S1:,} S1 entities and ground truth...", flush=True)
    s1_dict = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        count = 0
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                eid, name, addr, country = p[0], p[1], p[2], p[3]
                s1_dict[eid] = {
                    'entity_id': eid,
                    'business_name': name,
                    'business_address': addr,
                    'country': country,
                    'name_tokens': get_tokens(name),
                    'addr_parsed': parse_address_fast(addr),
                    'addr_nums': get_distinctive_numbers(addr)
                }
                count += 1
                if count >= TOTAL_S1:
                    break

    s1_ids = list(s1_dict.keys())
    train_s1_ids = set(s1_ids[:N_TRAIN_S1])
    val_s1_ids = set(s1_ids[N_TRAIN_S1:])

    gt_map = {}
    all_needed_cands = set()
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if p:
                sid = p[0]
                if sid in s1_dict:
                    m = [x.strip() for x in p[1].split(',') if x.strip() and x.strip() != 'nan'] if len(p) > 1 else []
                    gt_map[sid] = m
                    all_needed_cands.update(m)

    print(f"  Loaded {len(s1_dict):,} S1 entities in {time.time() - t0:.2f}s. Target matches: {len(all_needed_cands):,}", flush=True)

    # Fast-load S2 and S3 needed records + negative candidates
    t1 = time.time()
    print("[2/5] Fast-loading matching candidate records from S2 and S3...", flush=True)
    cands_dict = {}
    s2_remaining = {x for x in all_needed_cands if x.startswith('S2-')}
    s3_remaining = {x for x in all_needed_cands if x.startswith('S3-')}

    # Read S2 with O(1) removal
    with open('dataset/train/train_source2.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        bg_count = 0
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                cid, name, addr, country = p[0], p[1], p[2], p[3]
                if cid in s2_remaining:
                    s2_remaining.remove(cid)
                    cands_dict[cid] = {
                        'entity_id': cid,
                        'business_name': name,
                        'business_address': addr,
                        'country': country,
                        'name_tokens': get_tokens(name),
                        'addr_parsed': parse_address_fast(addr),
                        'addr_nums': get_distinctive_numbers(addr)
                    }
                elif bg_count < 15000:
                    bg_count += 1
                    cands_dict[cid] = {
                        'entity_id': cid,
                        'business_name': name,
                        'business_address': addr,
                        'country': country,
                        'name_tokens': get_tokens(name),
                        'addr_parsed': parse_address_fast(addr),
                        'addr_nums': get_distinctive_numbers(addr)
                    }
                if not s2_remaining and bg_count >= 15000:
                    break

    # Read S3 with O(1) removal
    with open('dataset/train/train_source3.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        bg_count = 0
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                cid, name, addr, country = p[0], p[1], p[2], p[3]
                if cid in s3_remaining:
                    s3_remaining.remove(cid)
                    cands_dict[cid] = {
                        'entity_id': cid,
                        'business_name': name,
                        'business_address': addr,
                        'country': country,
                        'name_tokens': get_tokens(name),
                        'addr_parsed': parse_address_fast(addr),
                        'addr_nums': get_distinctive_numbers(addr)
                    }
                elif bg_count < 15000:
                    bg_count += 1
                    cands_dict[cid] = {
                        'entity_id': cid,
                        'business_name': name,
                        'business_address': addr,
                        'country': country,
                        'name_tokens': get_tokens(name),
                        'addr_parsed': parse_address_fast(addr),
                        'addr_nums': get_distinctive_numbers(addr)
                    }
                if not s3_remaining and bg_count >= 15000:
                    break

    print(f"  Candidate pool ready: {len(cands_dict):,} records (Loaded in {time.time() - t1:.2f}s)", flush=True)

    # Build inverted index
    t2 = time.time()
    print("[3/5] Building inverted indices with strict frequency pruning...", flush=True)
    inv_name = {}
    inv_num = {}
    for cid, r in cands_dict.items():
        country = r['country']
        for tok in r['name_tokens']:
            inv_name.setdefault((country, tok), []).append(cid)
        for num in r['addr_nums']:
            inv_num.setdefault((country, num), []).append(cid)

    # Strict pruning: tokens <= 150, numbers <= 50
    pruned_inv_name = {k: v for k, v in inv_name.items() if len(v) <= 150}
    pruned_inv_num = {k: v for k, v in inv_num.items() if len(v) <= 50}
    print(f"  Inverted index built in {time.time() - t2:.2f}s (Tokens: {len(pruned_inv_name):,}, Numbers: {len(pruned_inv_num):,})", flush=True)

    # Candidate generation with top-K cap per S1
    t3 = time.time()
    print("  Generating candidate pairs (max 30 candidates per entity)...", flush=True)
    def get_cands(sid, max_k=30):
        r = s1_dict[sid]
        country = r['country']
        scores = {}
        for tok in r['name_tokens']:
            if (country, tok) in pruned_inv_name:
                for cid in pruned_inv_name[(country, tok)]:
                    scores[cid] = scores.get(cid, 0) + 2
        for num in r['addr_nums']:
            if (country, num) in pruned_inv_num:
                for cid in pruned_inv_num[(country, num)]:
                    scores[cid] = scores.get(cid, 0) + 3
        if not scores:
            return []
        sorted_cands = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        return sorted_cands[:max_k]

    train_pairs = []
    val_pairs = []

    for sid in train_s1_ids:
        cands = set(get_cands(sid, max_k=25))
        for mid in gt_map.get(sid, []):
            if mid in cands_dict:
                cands.add(mid)
        negatives = [c for c in cands if c not in gt_map.get(sid, [])]
        if len(negatives) > 15:
            negatives = random.sample(negatives, 15)
        final_cands = set(negatives) | (set(gt_map.get(sid, [])) & set(cands_dict.keys()))
        for cid in final_cands:
            train_pairs.append((sid, cid))

    val_true_matches = 0
    val_captured_matches = 0
    for sid in val_s1_ids:
        cands = get_cands(sid, max_k=30)
        true_set = set(gt_map.get(sid, [])) & set(cands_dict.keys())
        val_true_matches += len(true_set)
        val_captured_matches += len(true_set & set(cands))
        for cid in cands:
            val_pairs.append((sid, cid))

    print(f"  Pairs generated in {time.time() - t3:.2f}s:")
    print(f"  Train Pairs: {len(train_pairs):,} | Val Pairs: {len(val_pairs):,}", flush=True)
    if val_true_matches > 0:
        print(f"  Val Blocking Recall: {val_captured_matches}/{val_true_matches} ({val_captured_matches/val_true_matches*100:.2f}%)", flush=True)

    # Extract features
    t4 = time.time()
    print("[4/5] Extracting pairwise features for LightGBM training...", flush=True)
    X_train = np.array([compute_pairwise_features(s1_dict[sid], cands_dict[cid]) for sid, cid in train_pairs], dtype=np.float32)
    y_train = np.array([1 if cid in gt_map.get(sid, []) else 0 for sid, cid in train_pairs], dtype=np.int32)
    print(f"  Extracted {len(X_train):,} training features in {time.time() - t4:.2f}s. Positive links: {np.sum(y_train):,} ({np.mean(y_train)*100:.1f}%)", flush=True)

    # Train LightGBM model with all 12 cores
    t5 = time.time()
    print("  Fitting LightGBM on all 12 CPU cores...", flush=True)
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
    print(f"  Model trained in {time.time() - t5:.2f}s!", flush=True)

    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/matching_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("  Saved model artifact to artifacts/matching_model.pkl", flush=True)

    # Feature importances
    print("\n  --- Top Feature Importances ---", flush=True)
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    for col, v in imp.items():
        print(f"    {col:20s}: {v}", flush=True)

    # Evaluate on Validation set
    t6 = time.time()
    print("\n[5/5] Scoring Validation Set & Sweeping Decision Threshold...", flush=True)
    X_val = np.array([compute_pairwise_features(s1_dict[sid], cands_dict[cid]) for sid, cid in val_pairs], dtype=np.float32)
    val_probs = clf.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    print(f"  Validation features and probabilities computed in {time.time() - t6:.2f}s!", flush=True)

    val_gt = {sid: gt_map.get(sid, []) for sid in val_s1_ids}

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
    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = run_collective_eval(primary_t=t)
        f05 = compute_macro_f05(val_gt, preds)
        print(f"  Threshold {t:.2f} -> Validation Macro F0.5: {f05:.4f}", flush=True)
        if f05 > best_f05:
            best_f05 = f05
            best_t = t

    print("\n" + "=" * 65, flush=True)
    print(f"  BEST VALIDATION RESULT: Threshold = {best_t:.2f} | Macro F0.5 = {best_f05:.4f}", flush=True)
    print(f"  Total End-to-End Elapsed Time: {time.time() - t0:.2f} seconds", flush=True)
    print("=" * 65, flush=True)

    with open('artifacts/best_threshold.json', 'w') as f:
        import json
        json.dump({'best_threshold': best_t, 'val_f05': best_f05}, f)

if __name__ == '__main__':
    train_and_validate()
