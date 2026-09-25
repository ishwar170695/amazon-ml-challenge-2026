import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from business_entity_resolution import (
    generate_synthetic_challenge_data, normalize_text, extract_significant_tokens,
    generate_acronyms, parse_address_components, strip_legal_suffixes,
    build_blocking_candidates, extract_pairwise_features, collective_resolve, compute_macro_f05
)
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def run_iteration_002():
    print("=" * 80)
    print("  RUNNING ITERATION 002: ADVERSARIAL GENERATOR AUDIT & LOCALITY-HARDNESS TEST")
    print("=" * 80)
    
    df_s1, df_s2, df_s3, ground_truth = generate_synthetic_challenge_data(n_entities=1200)
    df_cands = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # 1. Audit for exact address leakage in France test set
    france_s1_df = df_s1[df_s1['country'] == 'France']
    france_s1_ids = set(france_s1_df['entity_id'])
    
    exact_leak_count = 0
    raw_cand_dict = df_cands.set_index('entity_id').to_dict('index')
    for _, row in france_s1_df.iterrows():
        sid = row['entity_id']
        s1_addr = row['business_address']
        for cid in ground_truth[sid]:
            c_addr = raw_cand_dict[cid]['business_address']
            if s1_addr == c_addr:
                exact_leak_count += 1
                
    print(f"Exact uncorrupted address matches in France ground-truth pairs: {exact_leak_count} (Target: 0)")
    
    # 2. Preprocess & extract features
    for df in [df_s1, df_cands]:
        df['norm_name'] = df['business_name'].apply(lambda x: normalize_text(x, is_address=False))
        df['norm_address'] = df['business_address'].apply(lambda x: normalize_text(x, is_address=True))
        df['sig_tokens'] = df['norm_name'].apply(extract_significant_tokens)
        df['acronyms'] = df['norm_name'].apply(generate_acronyms)
        df['parsed_addr'] = df['business_address'].apply(parse_address_components)
        df['stripped_name'] = df['norm_name'].apply(strip_legal_suffixes)
        
    all_names = list(df_s1['norm_name']) + list(df_cands['norm_name'])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(all_names)
    s1_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_s1.iterrows()}
    cand_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_cands.iterrows()}
    
    # 3. Blocking & candidate indexing
    pairs, total_true = build_blocking_candidates(df_s1, df_cands)
    flat_gt = set((sid, m) for sid, ml in ground_truth.items() for m in ml)
    blocking_recall = len(set(pairs).intersection(flat_gt)) / len(flat_gt)
    print(f"Total candidate pairs: {len(pairs)}, Blocking Recall: {blocking_recall*100:.2f}%")
    
    df_s1_dict = df_s1.set_index('entity_id').to_dict('index')
    df_cand_dict = df_cands.set_index('entity_id').to_dict('index')
    X = extract_pairwise_features(pairs, df_s1_dict, df_cand_dict, s1_tfidf_map, cand_tfidf_map)
    y = np.array([1 if p in flat_gt else 0 for p in pairs])
    
    # 4. Entity-level split: US & India -> Train/Val, France -> Held-out Test
    non_france = df_s1[df_s1['country'] != 'France']['entity_id'].values
    np.random.seed(42); np.random.shuffle(non_france)
    split = int(len(non_france) * 0.75)
    train_ids = set(non_france[:split])
    val_ids = set(non_france[split:])
    test_ids = france_s1_ids
    
    s1_idx = X.index.get_level_values('s1')
    X_train, y_train = X[s1_idx.isin(train_ids)], y[s1_idx.isin(train_ids)]
    X_val, y_val = X[s1_idx.isin(val_ids)], y[s1_idx.isin(val_ids)]
    X_test = X[s1_idx.isin(test_ids)]
    
    clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_gt = {sid: ground_truth[sid] for sid in val_ids}
    
    # Val threshold sweep
    best_t, best_val_f05 = 0.20, 0.0
    print("\n--- Validation Threshold Sweep (US/India) ---")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        coll_val = collective_resolve(X_val.index, val_probs, val_ids, primary_t=t, secondary_t=t+0.15, max_margin=0.25)
        f05, p, r = compute_macro_f05(val_gt, coll_val)
        print(f"  Thresh {t:.2f} | F0.5: {f05:.4f} | Prec: {p*100:5.1f}% | Rec: {r*100:5.1f}%")
        if f05 > best_val_f05:
            best_val_f05 = f05
            best_t = t
            
    print(f"\nOptimal Val Threshold: {best_t:.2f} (Val F0.5: {best_val_f05:.4f})")
    
    # Evaluate on held-out France test using the frozen val threshold
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_gt = {sid: ground_truth[sid] for sid in test_ids}
    coll_test = collective_resolve(X_test.index, test_probs, test_ids, primary_t=best_t, secondary_t=best_t+0.15, max_margin=0.25)
    test_f05, test_p, test_r = compute_macro_f05(test_gt, coll_test)
    
    # Accepted matches distribution
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
    
    print("\n" + "=" * 80)
    print(f"  HELD-OUT TEST RESULTS (FRANCE, Frozen Thresh {best_t:.2f})")
    print("=" * 80)
    print(f"  Macro F0.5:         {test_f05:.4f}")
    print(f"  Precision:          {test_p*100:.2f}%")
    print(f"  Recall:             {test_r*100:.2f}%")
    print(f"  Total Accepted:     {len(accepted_probs)}")
    print(f"  Min Accepted Prob:  {min_acc_p:.3f}")
    print(f"  Total False Merges: {len(fps)}")
    print(f"  Total Missed Links: {len(fns)}")
    
    # Feature importances
    importances = clf.feature_importances_
    feat_names = X.columns
    sorted_feats = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    total_imp = sum(importances)
    for feat, imp in sorted_feats[:5]:
        print(f"  - {feat:<24}: {imp/total_imp*100:.1f}%")
        
    # Sample False Positives & Negatives
    print("\nSampled False Negatives (Missed Matches):")
    for sid, cid, p in fns[:4]:
        r1, r2 = df_s1_dict[sid], df_cand_dict[cid]
        print(f"  • S1: [{r1['business_name']}] @ [{r1['business_address']}]")
        print(f"    Cand: [{r2['business_name']}] @ [{r2['business_address']}] (Prob: {p:.3f})\n")
        
    return {
        'val_thresh': best_t,
        'val_f05': best_val_f05,
        'test_f05': test_f05,
        'precision': test_p,
        'recall': test_r,
        'blocking_recall': blocking_recall,
        'candidate_count': len(pairs),
        'min_accepted_prob': min_acc_p,
        'total_accepted': len(accepted_probs),
        'exact_leaks': exact_leak_count,
        'n_fps': len(fps),
        'n_fns': len(fns)
    }

if __name__ == '__main__':
    res = run_iteration_002()
    # Save to ledger
    ledger_entry = {
        "iteration_id": "iter_002",
        "timestamp": "2026-09-25T12:54:00+05:30",
        "hypothesis": "Universal country-aware address perturbation and locality corruption across S2 and S3 eliminates exact-duplicate leakage, yielding a genuine adversarial test.",
        "change": "Replaced hardcoded .replace() with corrupt_street_address + multi-match S3 locality corruption.",
        "dataset": "synthetic_business_er_adversarial_v3",
        "split": "entity_level_us_in_train_fr_test",
        "blocking": "token_union_prefix_acronym",
        "candidate_count": res['candidate_count'],
        "blocking_recall": res['blocking_recall'],
        "model": "LightGBM",
        "features": "15_pairwise_features",
        "threshold": res['val_thresh'],
        "precision": res['precision'],
        "recall": res['recall'],
        "f05": res['test_f05'],
        "exact_leaks": res['exact_leaks'],
        "decision": "KEEP"
    }
    with open("experiments/ledger.jsonl", "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    print("Logged iteration 002 to experiments/ledger.jsonl")
