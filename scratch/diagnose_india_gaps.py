import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import re, unicodedata, anyascii
from run_pipeline import make_record, build_inverted_index, query_candidates

print("Loading 2,000 India S1 records...", flush=True)
s1_sample = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4 and p[3] == 'India':
            s1_sample[p[0]] = make_record(p[0], p[1], p[2], 'India')
            if len(s1_sample) >= 2000:
                break

print(f"Loaded {len(s1_sample)} S1 records. Loading GT...", flush=True)
gt_map = {}
needed_cands = set()
with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 2 and p[0] in s1_sample:
            m = [x.strip() for x in p[1].split(',') if x.strip() and x.strip() != 'nan']
            if m:
                gt_map[p[0]] = m
                needed_cands.update(m)

print(f"Found {len(gt_map)} S1 with GT ({len(needed_cands)} true matches needed). Loading candidates...", flush=True)
cand_dict = {}
for split_name in ['train_source2.tsv', 'train_source3.tsv']:
    with open(f'dataset/train/{split_name}', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4 and p[0] in needed_cands:
                cand_dict[p[0]] = make_record(p[0], p[1], p[2], 'India')
                if len(cand_dict) >= len(needed_cands):
                    break
    if len(cand_dict) >= len(needed_cands):
        break

print(f"Loaded {len(cand_dict)} / {len(needed_cands)} candidate records.", flush=True)

inv = build_inverted_index(cand_dict, prefix_limit=500, token_limit=500)

unblocked = []
blocked = 0
for s1_id, targets in gt_map.items():
    r1 = s1_sample[s1_id]
    cands = query_candidates(r1, inv)
    for t in targets:
        if t in cand_dict:
            if t in cands:
                blocked += 1
            else:
                unblocked.append((r1, cand_dict[t]))

total = blocked + len(unblocked)
print(f"\n==========================================")
print(f"India Sample Blocking Recall: {blocked}/{total} = {blocked/total*100:.2f}% (Missed: {len(unblocked)})")
print(f"==========================================\n")

print("--- UNBLOCKED INDIA PAIRS ---")
for i, (r1, r2) in enumerate(unblocked[:15]):
    print(f"[{i+1}] S1: {r1['norm_name']} | Addr: {r1['norm_address']}")
    print(f"     C : {r2['norm_name']} | Addr: {r2['norm_address']}")
    print(f"     S1 sig: {r1['sig_tokens']} | C sig: {r2['sig_tokens']}")
    print(f"     S1 PH : {r1['phonetic_tokens']} | C PH : {r2['phonetic_tokens']}")
    print(f"     S1 nums: {r1['addr_nums']} | C nums: {r2['addr_nums']}")
    print("-" * 60)
