import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import pickle
from scratch.diagnose_india_gaps import s1_sample, cand_dict, gt_map
from run_pipeline import query_candidates, build_inverted_index, extract_features_batch, csr_to_dict_list, collective_resolve

with open('artifacts/model_v3.pkl', 'rb') as f:
    art = pickle.load(f)
clf = art['model']
wvec, cvec = art['word_tfidf'], art['char_tfidf']

inv = build_inverted_index(cand_dict)
pairs = []
for sid, r in s1_sample.items():
    for cid in query_candidates(r, inv):
        pairs.append((sid, cid))

s1_uniq = list({s for s, _ in pairs})
c_uniq = list({c for _, c in pairs})
s1n = [s1_sample[s]['stripped_name'] for s in s1_uniq]
cn = [cand_dict[c]['stripped_name'] for c in c_uniq]

s1_wdict = dict(zip(s1_uniq, csr_to_dict_list(wvec.transform(s1n))))
s1_cdict = dict(zip(s1_uniq, csr_to_dict_list(cvec.transform(s1n))))
c_wdict = dict(zip(c_uniq, csr_to_dict_list(wvec.transform(cn))))
c_cdict = dict(zip(c_uniq, csr_to_dict_list(cvec.transform(cn))))

X = extract_features_batch(pairs, s1_sample, cand_dict, s1_wdict, c_wdict, s1_cdict, c_cdict)
probs = clf.predict_proba(X)[:, 1]

# Map probabilities by pair
pair_prob = {pair: p for pair, p in zip(pairs, probs)}

# Run collective resolution with current settings
preds = collective_resolve(pairs, probs, list(s1_sample.keys()), primary_threshold=0.88, secondary_threshold=0.88, pair_features=X, s1_dict=s1_sample)

# Analyze False Negatives
fn_pairs = []
for sid, targets in gt_map.items():
    predicted = set(preds.get(sid, []))
    for t in targets:
        if t in cand_dict and t not in predicted:
            p = pair_prob.get((sid, t), -1.0)
            fn_pairs.append((sid, t, p))

print(f"Total True Matches: {sum(len(v) for v in gt_map.values()):,}")
print(f"Total False Negatives: {len(fn_pairs):,}")

# Distribution of FN probabilities
p_not_blocked = sum(1 for _, _, p in fn_pairs if p == -1.0)
p_sub_50 = sum(1 for _, _, p in fn_pairs if 0.0 <= p < 0.50)
p_50_75 = sum(1 for _, _, p in fn_pairs if 0.50 <= p < 0.75)
p_75_88 = sum(1 for _, _, p in fn_pairs if 0.75 <= p < 0.88)
p_above_88 = sum(1 for _, _, p in fn_pairs if p >= 0.88)

print(f"  Not blocked           : {p_not_blocked}")
print(f"  Scored P < 0.50       : {p_sub_50}")
print(f"  Scored 0.50 <= P < 0.75: {p_50_75}")
print(f"  Scored 0.75 <= P < 0.88: {p_75_88}")
print(f"  Scored P >= 0.88      : {p_above_88} (filtered out by secondary guard/cap/winner)")

print("\n--- SAMPLE OF FN PAIRS SCORED 0.50 <= P < 0.88 ---")
for sid, cid, p in [x for x in fn_pairs if 0.50 <= x[2] < 0.88][:5]:
    r1 = s1_sample[sid]
    r2 = cand_dict[cid]
    print(f"P = {p:.4f}")
    print(f"  S1: {r1['norm_name']} | Addr: {r1['norm_address']}")
    print(f"  C : {r2['norm_name']} | Addr: {r2['norm_address']}")
    print("-" * 50)

print("\n--- SAMPLE OF FN PAIRS SCORED P >= 0.88 (SECONDARY GUARD FILTERED) ---")
for sid, cid, p in [x for x in fn_pairs if x[2] >= 0.88][:5]:
    r1 = s1_sample[sid]
    r2 = cand_dict[cid]
    print(f"P = {p:.4f}")
    print(f"  S1: {r1['norm_name']} | Addr: {r1['norm_address']}")
    print(f"  C : {r2['norm_name']} | Addr: {r2['norm_address']}")
    print("-" * 50)
