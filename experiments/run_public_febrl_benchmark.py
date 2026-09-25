import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import recordlinkage.datasets as ds
from business_entity_resolution import (
    normalize_text, extract_significant_tokens, generate_acronyms,
    parse_address_components, strip_legal_suffixes,
    build_blocking_candidates, extract_pairwise_features, collective_resolve, compute_macro_f05
)
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_format_febrl4():
    dfA, dfB = ds.load_febrl4()
    
    # Strictly remove cheat IDs: NO date_of_birth, NO soc_sec_id!
    def clean_str(val):
        if pd.isna(val):
            return ""
        s = str(val).strip()
        return "" if s == "nan" else s

    records_s1 = []
    for rec_id, row in dfA.iterrows():
        name = f"{clean_str(row['given_name'])} {clean_str(row['surname'])}".strip()
        num = clean_str(row['street_number'])
        a1 = clean_str(row['address_1'])
        a2 = clean_str(row['address_2'])
        sub = clean_str(row['suburb'])
        st = clean_str(row['state'])
        pc = clean_str(row['postcode'])
        addr_parts = [p for p in [f"{num} {a1}".strip(), a2, sub, f"{st} {pc}".strip()] if p]
        addr = ', '.join(addr_parts)
        records_s1.append({
            'entity_id': rec_id,
            'business_name': name,
            'business_address': addr,
            'country': 'AU'
        })

    records_s2 = []
    for rec_id, row in dfB.iterrows():
        name = f"{clean_str(row['given_name'])} {clean_str(row['surname'])}".strip()
        num = clean_str(row['street_number'])
        a1 = clean_str(row['address_1'])
        a2 = clean_str(row['address_2'])
        sub = clean_str(row['suburb'])
        st = clean_str(row['state'])
        pc = clean_str(row['postcode'])
        addr_parts = [p for p in [f"{num} {a1}".strip(), a2, sub, f"{st} {pc}".strip()] if p]
        addr = ', '.join(addr_parts)
        records_s2.append({
            'entity_id': rec_id,
            'business_name': name,
            'business_address': addr,
            'country': 'AU'
        })

    df_s1 = pd.DataFrame(records_s1)
    df_s2 = pd.DataFrame(records_s2)

    # Gold standard mapping
    # rec-1070-org in A matches rec-1070-dup-0 in B
    b_id_map = {rec_id.split('-dup-')[0]: rec_id for rec_id in df_s2['entity_id']}
    ground_truth = {}
    for sid in df_s1['entity_id']:
        base = sid.split('-org')[0]
        if base in b_id_map:
            ground_truth[sid] = [b_id_map[base]]
        else:
            ground_truth[sid] = []

    return df_s1, df_s2, ground_truth

def evaluate_pipeline(X, y, df_s1, df_cands, ground_truth, train_ids, val_ids, test_ids, do_hard_negative_mining=False):
    s1_idx = X.index.get_level_values('s1')
    X_train, y_train = X[s1_idx.isin(train_ids)], y[s1_idx.isin(train_ids)]
    X_val, y_val = X[s1_idx.isin(val_ids)], y[s1_idx.isin(val_ids)]
    X_test, y_test = X[s1_idx.isin(test_ids)], y[s1_idx.isin(test_ids)]

    clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)

    if do_hard_negative_mining:
        print("\n  [Active Learning] Mining Hard Negatives from Training Candidate Pool...")
        train_probs = clf.predict_proba(X_train)[:, 1]
        # Mine false alarms: non-matches that score high probability P in [0.20, 0.70]
        hard_neg_mask = (y_train == 0) & (train_probs >= 0.20)
        n_mined = hard_neg_mask.sum()
        print(f"  Mined {n_mined} hard negative pairs (P >= 0.20 on non-matches)")
        
        # Duplicate hard negatives with higher weight in training
        if n_mined > 0:
            X_hard = X_train[hard_neg_mask]
            y_hard = y_train[hard_neg_mask]
            X_train_aug = pd.concat([X_train, X_hard, X_hard], axis=0)
            y_train_aug = np.concatenate([y_train, y_hard, y_hard])
            clf.fit(X_train_aug, y_train_aug)
            print(f"  Retrained LightGBM with {len(X_train_aug):,} total training samples")

    # 1. Validation Sweep
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_gt = {sid: ground_truth[sid] for sid in val_ids}
    best_t, best_val_f05 = 0.20, 0.0
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        coll_val = collective_resolve(X_val.index, val_probs, val_ids, primary_t=t, secondary_t=t+0.15, max_margin=0.25)
        f05, p, r = compute_macro_f05(val_gt, coll_val)
        if f05 > best_val_f05:
            best_val_f05 = f05
            best_t = t

    # 2. Frozen Test Evaluation
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_gt = {sid: ground_truth[sid] for sid in test_ids}
    coll_test = collective_resolve(X_test.index, test_probs, test_ids, primary_t=best_t, secondary_t=best_t+0.15, max_margin=0.25)
    test_f05, test_p, test_r = compute_macro_f05(test_gt, coll_test)

    # Scrutiny
    test_pairs_list = list(X_test.index)
    accepted_probs = []
    fps = []
    fns = []
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

    min_acc_p = min(accepted_probs) if accepted_probs else 0.0

    return {
        'best_thresh': best_t,
        'val_f05': best_val_f05,
        'test_f05': test_f05,
        'precision': test_p,
        'recall': test_r,
        'n_accepted': len(accepted_probs),
        'min_accepted_prob': min_acc_p,
        'n_fps': len(fps),
        'n_fns': len(fns),
        'fps': fps,
        'fns': fns,
        'clf': clf
    }

def main():
    print("=" * 80)
    print("  INDEPENDENT PUBLIC BENCHMARK: FEBRL4 (10,000 REAL NOISY RECORDS)  ")
    print("  NO CHEAT IDs: date_of_birth and soc_sec_id 100% STRIPPED")
    print("=" * 80)

    df_s1, df_s2, ground_truth = load_and_format_febrl4()
    print(f"Loaded: Source 1 (Reference) = {len(df_s1):,} records, Source 2 (Candidates) = {len(df_s2):,} records")
    total_true_links = sum(len(m) for m in ground_truth.values())
    print(f"Total True Links: {total_true_links:,}")

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

    # Multi-key blocking
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
    print(f"Feature Matrix Shape: {X.shape}, True Links: {y.sum():,}, Negatives: {(y==0).sum():,}")

    # Entity-level 70 / 15 / 15 split
    all_s1_ids = df_s1['entity_id'].values
    np.random.seed(42); np.random.shuffle(all_s1_ids)
    n_train = int(len(all_s1_ids) * 0.70)
    n_val = int(len(all_s1_ids) * 0.15)
    train_ids = set(all_s1_ids[:n_train])
    val_ids = set(all_s1_ids[n_train:n_train+n_val])
    test_ids = set(all_s1_ids[n_train+n_val:])
    print(f"\nEntity Split: Train={len(train_ids):,}, Val={len(val_ids):,}, Held-out Test={len(test_ids):,}")

    # Step 1: Baseline Evaluation (Before Hard Negative Mining)
    print("\n" + "=" * 80)
    print("  RUN 1: BASELINE PIPELINE ON PUBLIC BENCHMARK")
    print("=" * 80)
    res_baseline = evaluate_pipeline(X, y, df_s1, df_cands=df_s2, ground_truth=ground_truth,
                                     train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                                     do_hard_negative_mining=False)

    print(f"  Optimal Val Thresh: {res_baseline['best_thresh']:.2f} (Val F0.5: {res_baseline['val_f05']:.4f})")
    print(f"  Held-out Test Macro F0.5: {res_baseline['test_f05']:.4f}")
    print(f"  Held-out Test Precision:  {res_baseline['precision']*100:.2f}%")
    print(f"  Held-out Test Recall:     {res_baseline['recall']*100:.2f}%")
    print(f"  Total False Merges:       {res_baseline['n_fps']}")
    print(f"  Total Missed Links:       {res_baseline['n_fns']}")

    # Step 2: HYP-005 - Active Hard Negative Mining
    print("\n" + "=" * 80)
    print("  RUN 2: HYP-005 ACTIVE HARD-NEGATIVE MINING")
    print("=" * 80)
    res_hardneg = evaluate_pipeline(X, y, df_s1, df_cands=df_s2, ground_truth=ground_truth,
                                    train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                                    do_hard_negative_mining=True)

    print(f"\n  [Post-Hard Negative Mining]")
    print(f"  Optimal Val Thresh: {res_hardneg['best_thresh']:.2f} (Val F0.5: {res_hardneg['val_f05']:.4f})")
    print(f"  Held-out Test Macro F0.5: {res_hardneg['test_f05']:.4f} (Delta: {res_hardneg['test_f05'] - res_baseline['test_f05']:+.4f})")
    print(f"  Held-out Test Precision:  {res_hardneg['precision']*100:.2f}% (Delta: {(res_hardneg['precision'] - res_baseline['precision'])*100:+.2f}%)")
    print(f"  Held-out Test Recall:     {res_hardneg['recall']*100:.2f}% (Delta: {(res_hardneg['recall'] - res_baseline['recall'])*100:+.2f}%)")
    print(f"  Total False Merges:       {res_hardneg['n_fps']}")
    print(f"  Total Missed Links:       {res_hardneg['n_fns']}")

    # Automated Error Taxonomy
    print("\n" + "=" * 80)
    print("  AUTOMATED ERROR TAXONOMY (HELD-OUT TEST)")
    print("=" * 80)
    fns = res_hardneg['fns']
    tax = {
        'street_number_conflict': 0,
        'typo_heavy_name_address': 0,
        'suburb_state_discrepancy': 0,
        'missing_fields': 0
    }
    for sid, cid, p in fns:
        r1, r2 = df_s1_dict[sid], df_cand_dict[cid]
        num1 = r1['parsed_addr']['number']
        num2 = r2['parsed_addr']['number']
        if num1 and num2 and num1 != num2:
            tax['street_number_conflict'] += 1
        elif not num1 or not num2:
            tax['missing_fields'] += 1
        elif r1['parsed_addr']['locality'] != r2['parsed_addr']['locality']:
            tax['suburb_state_discrepancy'] += 1
        else:
            tax['typo_heavy_name_address'] += 1

    total_errs = len(fns) if fns else 1
    for k, v in tax.items():
        print(f"  - {k:<28}: {v:3d} ({v/total_errs*100:5.1f}%)")

if __name__ == '__main__':
    main()
