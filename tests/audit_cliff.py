import pandas as pd
import numpy as np
from ablation_audit import make_data, new_parse_address
from business_entity_resolution import (
    normalize_text, extract_significant_tokens, generate_acronyms, strip_legal_suffixes,
    build_blocking_candidates, extract_pairwise_features, collective_resolve
)
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def audit_probability_cliff():
    df_s1, df_s2, df_s3, ground_truth = make_data(is_new_generator=True)
    df_cands = pd.concat([df_s2, df_s3], ignore_index=True)
    
    for df in [df_s1, df_cands]:
        df['norm_name'] = df['business_name'].apply(lambda x: normalize_text(x, is_address=False))
        df['norm_address'] = df['business_address'].apply(lambda x: normalize_text(x, is_address=True))
        df['sig_tokens'] = df['norm_name'].apply(extract_significant_tokens)
        df['acronyms'] = df['norm_name'].apply(generate_acronyms)
        df['parsed_addr'] = df['business_address'].apply(new_parse_address)
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
    test_pairs = list(X_test.index)
    test_gt = {sid: ground_truth[sid] for sid in test_ids}
    
    coll_030 = collective_resolve(X_test.index, test_probs, test_ids, primary_t=0.30, secondary_t=0.45, max_margin=0.25)
    accepted_pairs = set((sid, cid) for sid, clist in coll_030.items() for cid in clist)
    
    # Inspect all candidate pairs with probability between 0.10 and 0.748
    mid_band = []
    for i, (sid, cid) in enumerate(test_pairs):
        p = test_probs[i]
        is_gt = cid in test_gt[sid]
        is_acc = (sid, cid) in accepted_pairs
        mid_band.append((sid, cid, p, is_gt, is_acc))
        
    mid_band.sort(key=lambda x: x[2], reverse=True)
    
    print("=" * 80)
    print("  INSPECTING CANDIDATE PAIRS WITH PROBABILITY IN [0.15, 0.75] (REJECTED CLOSE CALLS)")
    print("=" * 80)
    
    in_range = [x for x in mid_band if 0.15 <= x[2] < 0.748]
    print(f"Total candidate pairs in [0.15, 0.748): {len(in_range)}")
    
    for sid, cid, p, is_gt, is_acc in in_range[:10]:
        r1 = df_s1_dict[sid]
        r2 = df_cand_dict[cid]
        status = "TRUE MATCH (FALSE NEGATIVE / REJECTED)" if is_gt else "TRUE NEGATIVE (CORRECT REJECT)"
        print(f"  • Prob: {p:.3f} | {status} | Accepted: {is_acc}")
        print(f"    S1:   [{r1['business_name']}] @ [{r1['business_address']}]")
        print(f"    Cand: [{r2['business_name']}] @ [{r2['business_address']}]\n")

audit_probability_cliff()
