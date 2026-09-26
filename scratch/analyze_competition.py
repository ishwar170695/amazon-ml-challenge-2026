import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import pickle
from scratch.diagnose_india_gaps import s1_sample, cand_dict, gt_map
from run_pipeline import query_candidates, build_inverted_index, extract_features_batch, csr_to_dict_list

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

cand_claims = {}
for (s1, c), p in zip(pairs, probs):
    if p >= 0.85:
        cand_claims.setdefault(c, []).append((s1, p))

multi_claim_cands = {c: claims for c, claims in cand_claims.items() if len(claims) > 1}
print(f"Total candidates with prob >= 0.85: {len(cand_claims):,}")
print(f"Candidates with claims from >1 S1 entities: {len(multi_claim_cands):,} ({len(multi_claim_cands)/len(cand_claims)*100:.2f}%)")

print("\n--- SAMPLE OF COMPETING CLAIMS ---")
for c, claims in list(multi_claim_cands.items())[:5]:
    c_rec = cand_dict[c]
    print(f"Candidate: {c_rec['norm_name']} | Addr: {c_rec['norm_address']}")
    for s1, p in claims:
        s1_rec = s1_sample[s1]
        is_true = (c in gt_map.get(s1, []))
        print(f"   S1 ({s1}) [TRUE={is_true}]: {s1_rec['norm_name']} | Addr: {s1_rec['norm_address']} | Prob={p:.4f}")
    print("-" * 50)
