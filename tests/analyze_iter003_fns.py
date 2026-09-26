import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from business_entity_resolution import (
    generate_synthetic_challenge_data, normalize_text, extract_significant_tokens,
    generate_acronyms, parse_address_components, strip_legal_suffixes,
    build_blocking_candidates, extract_pairwise_features, collective_resolve
)
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_fns():
    df_s1, df_s2, df_s3, ground_truth = generate_synthetic_challenge_data(n_entities=1200)
    df_cands = pd.concat([df_s2, df_s3], ignore_index=True)
    france_s1_df = df_s1[df_s1['country'] == 'France']
    france_s1_ids = set(france_s1_df['entity_id'])
    
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
    
    pairs, _ = build_blocking_candidates(df_s1, df_cands)
    flat_gt = set((sid, m) for sid, ml in ground_truth.items() for m in ml)
    pairs_set = set(pairs)
    
    df_s1_dict = df_s1.set_index('entity_id').to_dict('index')
    df_cand_dict = df_cands.set_index('entity_id').to_dict('index')
    X = extract_pairwise_features(pairs, df_s1_dict, df_cand_dict, s1_tfidf_map, cand_tfidf_map)
    y = np.array([1 if p in flat_gt else 0 for p in pairs])
    
    non_france = df_s1[df_s1['country'] != 'France']['entity_id'].values
    np.random.seed(42); np.random.shuffle(non_france)
    split = int(len(non_france) * 0.75)
    train_ids = set(non_france[:split])
    test_ids = france_s1_ids
    
    s1_idx = X.index.get_level_values('s1')
    X_train, y_train = X[s1_idx.isin(train_ids)], y[s1_idx.isin(train_ids)]
    X_test = X[s1_idx.isin(test_ids)]
    
    clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_gt = {sid: ground_truth[sid] for sid in test_ids}
    test_pairs_list = list(X_test.index)
    
    coll_test = collective_resolve(X_test.index, test_probs, test_ids, primary_t=0.15, secondary_t=0.30, max_margin=0.25)
    
    # Identify all false negatives
    fns = []
    for sid, gt_list in test_gt.items():
        pred_set = set(coll_test[sid])
        for cid in gt_list:
            if cid not in pred_set:
                in_blocking = (sid, cid) in pairs_set
                prob = test_probs[test_pairs_list.index((sid, cid))] if (sid, cid) in test_pairs_list else 0.0
                fns.append((sid, cid, in_blocking, prob))
                
    print("=" * 80)
    print(f"  DETAILED ANALYSIS OF ALL {len(fns)} FALSE NEGATIVES (MISSED MATCHES)")
    print("=" * 80)
    
    # 1. Blocking vs Classifier breakdown
    blocking_misses = [x for x in fns if not x[2]]
    classifier_misses = [x for x in fns if x[2]]
    print(f"\n1. Origin Breakdown:")
    print(f"   - Missed by Blocking:   {len(blocking_misses)} ({len(blocking_misses)/len(fns)*100:.1f}%)")
    print(f"   - Filtered by Model:    {len(classifier_misses)} ({len(classifier_misses)/len(fns)*100:.1f}%)")
    
    # 2. Probability distribution of classifier misses
    prob_0_05 = [x for x in classifier_misses if x[3] < 0.05]
    prob_05_10 = [x for x in classifier_misses if 0.05 <= x[3] < 0.10]
    prob_10_15 = [x for x in classifier_misses if 0.10 <= x[3] < 0.15]
    print(f"\n2. Model Probability Distribution for Classifier Misses:")
    print(f"   - Prob < 0.05 (near zero):        {len(prob_0_05)}")
    print(f"   - Prob 0.05 - 0.10 (low):         {len(prob_05_10)}")
    print(f"   - Prob 0.10 - 0.15 (near thresh): {len(prob_10_15)}")
    
    # 3. Categorization of structural characteristics
    cats = {
        'single_token_brand_and_landmark_addr': 0,
        'acronym_and_landmark_addr': 0,
        'typo_reorder_and_landmark_addr': 0,
        'other': 0
    }
    
    samples_by_cat = {k: [] for k in cats}
    
    for sid, cid, in_b, p in fns:
        r1, r2 = df_s1_dict[sid], df_cand_dict[cid]
        n1, n2 = r1['norm_name'], r2['norm_name']
        a1, a2 = r1['norm_address'], r2['norm_address']
        
        words2 = n2.split()
        if len(words2) == 1 and ('pres de la' in a2 or 'en face' in a2 or 'avenue victor' in a2):
            cats['single_token_brand_and_landmark_addr'] += 1
            if len(samples_by_cat['single_token_brand_and_landmark_addr']) < 3:
                samples_by_cat['single_token_brand_and_landmark_addr'].append((sid, cid, p, r1, r2))
        elif len(words2) == 1 and len(words2[0]) <= 4 and words2[0].isalpha():
            cats['acronym_and_landmark_addr'] += 1
            if len(samples_by_cat['acronym_and_landmark_addr']) < 3:
                samples_by_cat['acronym_and_landmark_addr'].append((sid, cid, p, r1, r2))
        elif ('pres de la' in a2 or 'en face' in a2 or 'avenue victor' in a2):
            cats['typo_reorder_and_landmark_addr'] += 1
            if len(samples_by_cat['typo_reorder_and_landmark_addr']) < 3:
                samples_by_cat['typo_reorder_and_landmark_addr'].append((sid, cid, p, r1, r2))
        else:
            cats['other'] += 1
            if len(samples_by_cat['other']) < 3:
                samples_by_cat['other'].append((sid, cid, p, r1, r2))
                
    print(f"\n3. Structural Category Breakdown of the 135 Misses:")
    for c, count in cats.items():
        print(f"   - {c:<38}: {count} ({count/len(fns)*100:.1f}%)")
        
    print("\n" + "=" * 80)
    print("  EXEMPLAR SAMPLES ACROSS MAIN FN CATEGORIES")
    print("=" * 80)
    for c, smp_list in samples_by_cat.items():
        if not smp_list:
            continue
        print(f"\nCategory: [{c.upper()}]")
        for sid, cid, p, r1, r2 in smp_list:
            print(f"  • S1:   [{r1['business_name']}] @ [{r1['business_address']}]")
            print(f"    Cand: [{r2['business_name']}] @ [{r2['business_address']}]")
            print(f"    Model Prob: {p:.3f} | Overlap: name_jaccard={len(set(r1['norm_name'].split()) & set(r2['norm_name'].split()))}\n")

if __name__ == '__main__':
    analyze_fns()
