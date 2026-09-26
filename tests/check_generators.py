import os, sys
import pandas as pd
import numpy as np
from ablation_audit import make_data, new_parse_address, old_parse_address
from business_entity_resolution import (
    normalize_text, extract_significant_tokens, generate_acronyms, strip_legal_suffixes,
    build_blocking_candidates, extract_pairwise_features, collective_resolve, compute_macro_f05
)
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def inspect_errors(is_new_generator, name):
    df_s1, df_s2, df_s3, ground_truth = make_data(is_new_generator=is_new_generator)
    df_cands = pd.concat([df_s2, df_s3], ignore_index=True)
    parser_fn = new_parse_address
    
    for df in [df_s1, df_cands]:
        df['norm_name'] = df['business_name'].apply(lambda x: normalize_text(x, is_address=False))
        df['norm_address'] = df['business_address'].apply(lambda x: normalize_text(x, is_address=True))
        df['sig_tokens'] = df['norm_name'].apply(extract_significant_tokens)
        df['acronyms'] = df['norm_name'].apply(generate_acronyms)
        df['parsed_addr'] = df['business_address'].apply(parser_fn)
        df['stripped_name'] = df['norm_name'].apply(strip_legal_suffixes)
        
    all_names = list(df_s1['norm_name']) + list(df_cands['norm_name'])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(all_names)
    s1_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_s1.iterrows()}
    cand_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_cands.iterrows()}
    
    pairs, _ = build_blocking_candidates(df_s1, df_cands)
    df_s1_dict = df_s1.set_index('entity_id').to_dict('index')
    df_cand_dict = df_cands.set_index('entity_id').to_dict('index')
    X = extract_pairwise_features(pairs, df_s1_dict, df_cand_dict, s1_tfidf_map, cand_tfidf_map)
    flat_gt = set((sid, m) for sid, ml in ground_truth.items() for m in ml)
    y = np.array([1 if p in flat_gt else 0 for p in pairs])
    
    france_s1 = set(df_s1[df_s1['country'] == 'France']['entity_id'])
    non_france = df_s1[df_s1['country'] != 'France']['entity_id'].values
    np.random.seed(42); np.random.shuffle(non_france)
    split = int(len(non_france)*0.75)
    train_ids = set(non_france[:split])
    test_ids = france_s1
    
    s1_idx = X.index.get_level_values('s1')
    X_train, y_train = X[s1_idx.isin(train_ids)], y[s1_idx.isin(train_ids)]
    X_test = X[s1_idx.isin(test_ids)]
    
    clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_gt = {sid: ground_truth[sid] for sid in test_ids}
    coll_test = collective_resolve(X_test.index, test_probs, test_ids, primary_t=0.20, secondary_t=0.35, max_margin=0.25)
    f05, p, r = compute_macro_f05(test_gt, coll_test)
    
    print(f"\n--- {name} (Threshold 0.20) ---")
    print(f"Macro F0.5: {f05:.4f}, Precision: {p*100:.2f}%, Recall: {r*100:.2f}%")
    
    fps = []
    test_pairs_list = list(X_test.index)
    for sid, m_list in coll_test.items():
        for cid in m_list:
            if cid not in test_gt[sid]:
                idx = test_pairs_list.index((sid, cid))
                fps.append((sid, cid, test_probs[idx]))
                
    print(f"Total False Positives: {len(fps)}")
    for sid, cid, prob in fps[:5]:
        print(f"  FP: S1={sid} [{df_s1_dict[sid]['business_name']}] @ [{df_s1_dict[sid]['business_address']}]")
        print(f"      Cand={cid} [{df_cand_dict[cid]['business_name']}] @ [{df_cand_dict[cid]['business_address']}] (Prob: {prob:.3f})")
        
    total_true_links = sum(len(m) for m in test_gt.values())
    total_found_links = sum(len([c for c in m if c in test_gt[sid]]) for sid, m in coll_test.items())
    print(f"Ground truth links: {total_true_links}, Found true links: {total_found_links}")

inspect_errors(False, "OLD GENERATOR")
inspect_errors(True, "NEW GENERATOR")
