import os
import warnings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import recordlinkage as rl
from recordlinkage.datasets import load_febrl4
from recordlinkage.preprocessing import phonetic
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score

def run_febrl_pipeline():
    print("=" * 65)
    print("  RIGOROUS ENTITY RESOLUTION PIPELINE (NO LEAKAGE, NO CHEAT IDs)  ")
    print("=" * 65)

    # -----------------------------------------------------------------
    # Step 1: Simulate Unstructured Business Text (Name, Address, State)
    # We explicitly EXCLUDE date_of_birth and soc_sec_id to avoid cheat IDs.
    # -----------------------------------------------------------------
    dfA, dfB, true_links = load_febrl4(return_links=True)

    for df in [dfA, dfB]:
        # Synthesize noisy raw business name & address fields
        df['name'] = (df['given_name'].fillna('') + ' ' + df['surname'].fillna('')).str.lower().str.strip()
        df['address'] = (df['street_number'].fillna('') + ' ' + df['address_1'].fillna('') + ' ' + df['suburb'].fillna('')).str.lower().str.strip()
        df['state'] = df['state'].fillna('').str.lower().str.strip()
        
        # Derived blocking keys from raw text tokens
        words = df['name'].str.split()
        df['w1'] = words.str[0].fillna('')
        df['w2'] = words.str[1].fillna('')
        df['w1_soundex'] = phonetic(df['w1'], 'soundex')
        df['w2_soundex'] = phonetic(df['w2'], 'soundex')
        df['name_prefix3'] = df['name'].str[:3]
        df['locality_key'] = df['suburb'].astype(str).str.lower().str.strip()

    print(f"Loaded: dfA={len(dfA)}, dfB={len(dfB)}, Ground-truth links={len(true_links)}")
    print("Excluded fields: ['soc_sec_id', 'date_of_birth'] (zero identifier leakage)")

    # -----------------------------------------------------------------
    # Step 2: Realistic Multi-Key Blocking (Candidate Generation)
    # Union of phonetic & token-prefix rules derived from noisy text
    # -----------------------------------------------------------------
    indexer = rl.Index()
    indexer.block('w1_soundex')
    indexer.block('w2_soundex')
    indexer.block('name_prefix3')
    indexer.block('locality_key')
    candidate_pairs = indexer.index(dfA, dfB)

    reduction = 100 * (1 - len(candidate_pairs) / (len(dfA) * len(dfB)))
    captured_links = len(true_links.intersection(candidate_pairs))
    blocking_recall = captured_links / len(true_links)
    print(f"\nCandidates generated: {len(candidate_pairs):,} (Reduction: {reduction:.2f}%)")
    print(f"Blocking Recall: {blocking_recall*100:.2f}% ({captured_links}/{len(true_links)} true links captured)")

    # -----------------------------------------------------------------
    # Step 3: Pure String / Text Similarity Features
    # -----------------------------------------------------------------
    compare = rl.Compare()
    compare.string('name', 'name', method='jarowinkler', threshold=0.6, label='name_jw')
    compare.string('name', 'name', method='levenshtein', threshold=0.5, label='name_lev')
    compare.string('address', 'address', method='jarowinkler', threshold=0.6, label='address_jw')
    compare.string('address', 'address', method='levenshtein', threshold=0.5, label='address_lev')
    compare.string('locality_key', 'locality_key', method='jarowinkler', threshold=0.7, label='locality_jw')
    compare.exact('state', 'state', label='state_exact')
    compare.exact('w1', 'w1', label='first_token_exact')
    compare.exact('w1_soundex', 'w1_soundex', label='phonetic_token_match')

    features = compare.compute(candidate_pairs, dfA, dfB)
    labels = pd.Series(0, index=candidate_pairs, dtype=int)
    labels.loc[candidate_pairs.intersection(true_links)] = 1

    # -----------------------------------------------------------------
    # Step 4: Leakage-Free Entity-Level Split (Train 60% / Val 20% / Test 20%)
    # S1 entities are partitioned so no entity spans across splits.
    # -----------------------------------------------------------------
    np.random.seed(42)
    s1_entities = np.array(dfA.index)
    np.random.shuffle(s1_entities)
    n = len(s1_entities)
    train_ids = set(s1_entities[:int(0.6 * n)])
    val_ids   = set(s1_entities[int(0.6 * n):int(0.8 * n)])
    test_ids  = set(s1_entities[int(0.8 * n):])

    s1_index = features.index.get_level_values(0)
    train_mask = s1_index.isin(train_ids)
    val_mask   = s1_index.isin(val_ids)
    test_mask  = s1_index.isin(test_ids)

    X_train, y_train = features[train_mask], labels[train_mask]
    X_val, y_val     = features[val_mask], labels[val_mask]
    X_test, y_test   = features[test_mask], labels[test_mask]

    # -----------------------------------------------------------------
    # Step 5: Train Matcher (LightGBM)
    # -----------------------------------------------------------------
    clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # Step 6: Tune Threshold on Validation Set specifically for F0.5
    # -----------------------------------------------------------------
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_thresh, best_val_f05 = 0.5, 0.0
    for t in np.linspace(0.1, 0.95, 86):
        preds = (val_probs >= t).astype(int)
        score = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if score > best_val_f05:
            best_val_f05, best_thresh = score, t

    print(f"\nOptimal Decision Threshold (selected on Val for F0.5): {best_thresh:.3f}")
    print(f"Validation F0.5: {best_val_f05:.4f}")

    # -----------------------------------------------------------------
    # Step 7: Unbiased Evaluation on Strictly HELD-OUT Test Set
    # -----------------------------------------------------------------
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)

    test_p = precision_score(y_test, test_preds, zero_division=0)
    test_r = recall_score(y_test, test_preds, zero_division=0)
    test_f05 = fbeta_score(y_test, test_preds, beta=0.5, zero_division=0)

    print("\n" + "-" * 40)
    print("  HELD-OUT TEST SET METRICS (UNBIASED)")
    print("-" * 40)
    print(f"  Precision: {test_p*100:.2f}%  (penalizes false merges)")
    print(f"  Recall:    {test_r*100:.2f}%")
    print(f"  F0.5:      {test_f05:.4f}")
    print("-" * 40)

    # -----------------------------------------------------------------
    # Step 8: End-to-End Resolution on Held-Out Test Entities
    # Handles singletons (0 matches) and one-to-many natively
    # -----------------------------------------------------------------
    test_matches = X_test.index[test_preds == 1]

    test_entity_map = {sid: [] for sid in test_ids}
    for s1_id, s2_id in test_matches:
        test_entity_map[s1_id].append(s2_id)

    singletons = sum(1 for v in test_entity_map.values() if len(v) == 0)
    matched = sum(1 for v in test_entity_map.values() if len(v) >= 1)
    print(f"\nHeld-Out Entities ({len(test_ids)} total):")
    print(f"  - Matched entities:     {matched}")
    print(f"  - Singletons predicted: {singletons}")

if __name__ == '__main__':
    run_febrl_pipeline()
