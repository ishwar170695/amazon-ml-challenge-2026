import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from business_entity_resolution import (
    normalize_text, extract_significant_tokens, generate_acronyms,
    parse_address_components, strip_legal_suffixes,
    build_blocking_candidates, extract_pairwise_features, collective_resolve, compute_macro_f05
)
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def load_fodors_zagat():
    data_dir = "data/fodors_zagat"
    dfA = pd.read_csv(f"{data_dir}/tableA.csv")
    dfB = pd.read_csv(f"{data_dir}/tableB.csv")
    
    # Clean text columns
    def clean_val(x):
        if pd.isna(x):
            return ""
        s = str(x).strip("` '\"\\")
        s = s.replace("\\ 's", "'s").replace("\\ '", "'").strip()
        return s

    s1_records = []
    for _, row in dfA.iterrows():
        name = clean_val(row['name'])
        addr = clean_val(row['addr'])
        city = clean_val(row['city'])
        full_addr = f"{addr}, {city}" if city else addr
        s1_records.append({
            'entity_id': f"FODORS_{row['id']}",
            'business_name': name,
            'business_address': full_addr,
            'country': 'US'
        })

    s2_records = []
    for _, row in dfB.iterrows():
        name = clean_val(row['name'])
        addr = clean_val(row['addr'])
        city = clean_val(row['city'])
        full_addr = f"{addr}, {city}" if city else addr
        s2_records.append({
            'entity_id': f"ZAGAT_{row['id']}",
            'business_name': name,
            'business_address': full_addr,
            'country': 'US'
        })

    df_s1 = pd.DataFrame(s1_records)
    df_s2 = pd.DataFrame(s2_records)

    # Load all ground truth pairs from train, valid, test
    gt_pairs = set()
    for split_file in ['train.csv', 'valid.csv', 'test.csv']:
        split_df = pd.read_csv(f"{data_dir}/{split_file}")
        for _, row in split_df[split_df['label'] == 1].iterrows():
            gt_pairs.add((f"FODORS_{row['ltable_id']}", f"ZAGAT_{row['rtable_id']}"))

    ground_truth = {sid: [] for sid in df_s1['entity_id']}
    for sid, cid in gt_pairs:
        if sid in ground_truth:
            ground_truth[sid].append(cid)

    return df_s1, df_s2, ground_truth

def main():
    print("=" * 80)
    print("  REAL-WORLD CORPORATE BENCHMARK: FODORS-ZAGAT RESTAURANT & CHAIN MATCHING  ")
    print("  Real multi-location chains, branch conflicts, noisy addresses, and singletons")
    print("=" * 80)

    df_s1, df_s2, ground_truth = load_fodors_zagat()
    total_true = sum(len(m) for m in ground_truth.values())
    n_singletons = sum(1 for m in ground_truth.values() if len(m) == 0)
    print(f"Source 1 (Fodors): {len(df_s1)} businesses ({n_singletons} Singletons, {len(df_s1)-n_singletons} with matches)")
    print(f"Source 2 (Zagats): {len(df_s2)} businesses")
    print(f"Total True Links:  {total_true}")

    # Normalization & address parsing
    print("\n[1/4] Normalizing Names, Address Components, and Acronyms...")
    for df in [df_s1, df_s2]:
        df['norm_name'] = df['business_name'].apply(lambda x: normalize_text(x, is_address=False))
        df['norm_address'] = df['business_address'].apply(lambda x: normalize_text(x, is_address=True))
        df['sig_tokens'] = df['norm_name'].apply(extract_significant_tokens)
        df['acronyms'] = df['norm_name'].apply(generate_acronyms)
        df['parsed_addr'] = df['business_address'].apply(parse_address_components)
        df['stripped_name'] = df['norm_name'].apply(strip_legal_suffixes)

    all_names = list(df_s1['norm_name']) + list(df_s2['norm_name'])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(all_names)
    s1_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_s1.iterrows()}
    cand_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_s2.iterrows()}

    # Candidate Blocking
    print("\n[2/4] Multi-Key Inverted Index Candidate Generation...")
    pairs, _ = build_blocking_candidates(df_s1, df_s2)
    flat_gt = set((sid, m) for sid, ml in ground_truth.items() for m in ml)
    blocking_recall = len(set(pairs).intersection(flat_gt)) / len(flat_gt)
    reduction = 1.0 - (len(pairs) / (len(df_s1) * len(df_s2)))
    print(f"Candidate Pairs Generated: {len(pairs):,} (Reduction: {reduction*100:.2f}%)")
    print(f"Blocking Recall:           {blocking_recall*100:.2f}% ({len(set(pairs).intersection(flat_gt))}/{len(flat_gt)})")

    # Feature extraction
    print("\n[3/4] Extracting 15 Decomposed Address & Name Features...")
    df_s1_dict = df_s1.set_index('entity_id').to_dict('index')
    df_cand_dict = df_s2.set_index('entity_id').to_dict('index')
    X = extract_pairwise_features(pairs, df_s1_dict, df_cand_dict, s1_tfidf_map, cand_tfidf_map)
    y = np.array([1 if p in flat_gt else 0 for p in pairs])
    print(f"Feature Matrix Shape: {X.shape}, True Links: {y.sum():,}, Hard Negatives: {(y==0).sum():,}")

    # Entity-level 60 / 20 / 20 split
    all_s1_ids = df_s1['entity_id'].values
    np.random.seed(42); np.random.shuffle(all_s1_ids)
    n_train = int(len(all_s1_ids) * 0.60)
    n_val = int(len(all_s1_ids) * 0.20)
    train_ids = set(all_s1_ids[:n_train])
    val_ids = set(all_s1_ids[n_train:n_train+n_val])
    test_ids = set(all_s1_ids[n_train+n_val:])
    print(f"\nEntity Split: Train={len(train_ids)}, Val={len(val_ids)}, Held-out Test={len(test_ids)}")

    s1_idx = X.index.get_level_values('s1')
    X_train, y_train = X[s1_idx.isin(train_ids)], y[s1_idx.isin(train_ids)]
    X_val, y_val = X[s1_idx.isin(val_ids)], y[s1_idx.isin(val_ids)]
    X_test, y_test = X[s1_idx.isin(test_ids)], y[s1_idx.isin(test_ids)]

    clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)

    # Active Hard Negative Mining Check on Corporate Chains
    print("\n[Active Learning Check on Corporate Candidates]")
    train_probs = clf.predict_proba(X_train)[:, 1]
    hard_negs = (y_train == 0) & (train_probs >= 0.15)
    print(f"Hard negative branch/chain candidates mined in training (P >= 0.15): {hard_negs.sum()}")
    if hard_negs.sum() > 0:
        X_hard = X_train[hard_negs]
        y_hard = y_train[hard_negs]
        X_train_aug = pd.concat([X_train, X_hard, X_hard], axis=0)
        y_train_aug = np.concatenate([y_train, y_hard, y_hard])
        clf.fit(X_train_aug, y_train_aug)
        print(f"Retrained LightGBM with {len(X_train_aug)} samples augmented with mined hard negatives")

    # Validation sweep
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_gt = {sid: ground_truth[sid] for sid in val_ids}
    best_t, best_val_f05 = 0.20, 0.0
    print("\n--- Validation Sweep ---")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        coll_val = collective_resolve(X_val.index, val_probs, val_ids, primary_t=t, secondary_t=t+0.15, max_margin=0.25)
        f05, p, r = compute_macro_f05(val_gt, coll_val)
        print(f"  Thresh {t:.2f} | F0.5: {f05:.4f} | Prec: {p*100:5.1f}% | Rec: {r*100:5.1f}%")
        if f05 > best_val_f05:
            best_val_f05 = f05
            best_t = t

    # Held-out Test Evaluation
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_gt = {sid: ground_truth[sid] for sid in test_ids}
    coll_test = collective_resolve(X_test.index, test_probs, test_ids, primary_t=best_t, secondary_t=best_t+0.15, max_margin=0.25)
    test_f05, test_p, test_r = compute_macro_f05(test_gt, coll_test)

    test_pairs_list = list(X_test.index)
    accepted_probs = []
    fps, fns = [], []
    for sid, m_list in coll_test.items():
        for cid in m_list:
            idx = test_pairs_list.index((sid, cid))
            p = test_probs[idx]
            accepted_probs.append(p)
            if cid not in test_gt[sid]:
                fps.append((sid, cid, p))

    for sid, gt_list in test_gt.items():
        pred_set = set(coll_test[sid])
        for cid in gt_list:
            if cid not in pred_set:
                p = test_probs[test_pairs_list.index((sid, cid))] if (sid, cid) in test_pairs_list else 0.0
                fns.append((sid, cid, p))

    print("\n" + "=" * 80)
    print(f"  HELD-OUT TEST RESULTS (FODORS-ZAGAT, Frozen Thresh {best_t:.2f})")
    print("=" * 80)
    print(f"  Macro F0.5:         {test_f05:.4f}")
    print(f"  Precision:          {test_p*100:.2f}%")
    print(f"  Recall:             {test_r*100:.2f}%")
    print(f"  Total Accepted:     {len(accepted_probs)}")
    print(f"  Total False Merges: {len(fps)}")
    print(f"  Total Missed Links: {len(fns)}")

    if fps:
        print("\nFalse Positives (Incorrect Merges):")
        for sid, cid, p in fps[:3]:
            r1, r2 = df_s1_dict[sid], df_cand_dict[cid]
            print(f"  • S1: [{r1['business_name']}] @ [{r1['business_address']}]")
            print(f"    Cand: [{r2['business_name']}] @ [{r2['business_address']}] (Prob: {p:.3f})")

    if fns:
        print("\nFalse Negatives (Missed Matches):")
        for sid, cid, p in fns[:3]:
            r1, r2 = df_s1_dict[sid], df_cand_dict[cid]
            print(f"  • S1: [{r1['business_name']}] @ [{r1['business_address']}]")
            print(f"    Cand: [{r2['business_name']}] @ [{r2['business_address']}] (Prob: {p:.3f})")

if __name__ == '__main__':
    main()
