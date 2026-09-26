import pandas as pd
import numpy as np
from ablation_audit import make_data, new_parse_address, old_parse_address
from business_entity_resolution import (
    normalize_text, extract_significant_tokens, generate_acronyms, strip_legal_suffixes,
    build_blocking_candidates, extract_pairwise_features, collective_resolve
)
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def get_links(is_new_generator):
    df_s1, df_s2, df_s3, ground_truth = make_data(is_new_generator=is_new_generator)
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
    coll = collective_resolve(X_test.index, test_probs, test_ids, primary_t=0.20, secondary_t=0.35, max_margin=0.25)
    return coll, {sid: ground_truth[sid] for sid in test_ids}, df_s1_dict, df_cand_dict, test_probs, list(X_test.index)

coll_old, gt_old, s1_old, cand_old, probs_old, pairs_old = get_links(False)
coll_new, gt_new, s1_new, cand_new, probs_new, pairs_new = get_links(True)

print("Difference between Old and New Generator predictions:")
for sid in gt_old:
    m_old = set(coll_old[sid])
    m_new = set(coll_new[sid])
    if m_new != m_old:
        print(f"S1 {sid}: Old={m_old} vs New={m_new} (GT={gt_old[sid]})")
        print(f"  Old S1: {s1_old[sid]['business_name']} | {s1_old[sid]['business_address']}")
        print(f"  New S1: {s1_new[sid]['business_name']} | {s1_new[sid]['business_address']}")
        for cid in (m_new - m_old):
            idx_old = pairs_old.index((sid, cid)) if (sid, cid) in pairs_old else None
            idx_new = pairs_new.index((sid, cid)) if (sid, cid) in pairs_new else None
            p_old = probs_old[idx_old] if idx_old is not None else 0.0
            p_new = probs_new[idx_new] if idx_new is not None else 0.0
            print(f"  Candidate {cid}:")
            print(f"    Old Cand: {cand_old[cid]['business_name']} | {cand_old[cid]['business_address']} -> Prob: {p_old:.3f}")
            print(f"    New Cand: {cand_new[cid]['business_name']} | {cand_new[cid]['business_address']} -> Prob: {p_new:.3f}")
        break
