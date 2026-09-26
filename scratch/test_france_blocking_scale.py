import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import random, re
from run_pipeline import make_record, build_inverted_index, query_candidates, normalize_text

random.seed(42)

print("Loading 500 real French S1 records from test_source1.tsv...", flush=True)
fr_s1_records = []
with open('dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4 and p[3] == 'France':
            fr_s1_records.append((p[0], p[1], p[2] if p[2] != 'nan' else ''))
            if len(fr_s1_records) >= 500:
                break

print(f"Loaded {len(fr_s1_records)} French reference records.", flush=True)

# Generate synthetic candidate variations matching the competition's open-set corruption patterns
french_abbr = {
    'rue': 'r', 'boulevard': 'bd', 'avenue': 'av', 'impasse': 'imp', 'chemin': 'che', 'route': 'rte',
    'allee': 'all', 'place': 'pl', 'faubourg': 'fg', 'saint': 'st', 'sainte': 'ste'
}
french_legal = ['sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'gie', 'earl', 'gaec']

synth_cands = {}
synth_gt = {}

for sid, name, addr in fr_s1_records:
    # Generate 1 to 3 realistic candidate variants for each S1 entity
    n_variants = random.choice([1, 2, 3])
    matched_ids = []
    
    for v_idx in range(n_variants):
        cid = f"SYNTH_C_{sid}_{v_idx}"
        
        # Corrupt name: add or drop legal suffix, reorder, or slight typo
        c_name = name
        if random.random() < 0.6:
            # Append or prepend French legal form
            c_name = f"{name} {random.choice(french_legal).upper()}"
        elif random.random() < 0.3:
            # Drop words or add & Fils / Cie
            c_name = f"{name} & Fils"
            
        # Corrupt address: abbreviate street type, drop or format CEDEX, alter number formatting
        c_addr = addr
        for full_w, ab in french_abbr.items():
            if re.search(rf'\b{full_w}\b', c_addr, re.I):
                c_addr = re.sub(rf'\b{full_w}\b', ab, c_addr, flags=re.I)
                break
                
        if random.random() < 0.4:
            # Add CEDEX or region
            c_addr = f"{c_addr}, France"
            
        synth_cands[cid] = make_record(cid, c_name, c_addr, 'France')
        matched_ids.append(cid)
        
    synth_gt[sid] = matched_ids

print(f"Generated {len(synth_cands)} synthetic French candidate records across {len(synth_gt)} entities.", flush=True)

# Also load 10,000 real French test distractors from test_source2.tsv to test index under realistic noise
print("Loading 10,000 background French test distractors...", flush=True)
with open('dataset/test/test_source2.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    cnt = 0
    for line in f:
        p = line.strip().split('\t')
        if len(p) >= 4 and p[3] == 'France':
            synth_cands[p[0]] = make_record(p[0], p[1], p[2] if p[2] != 'nan' else '', 'France')
            cnt += 1
            if cnt >= 10000:
                break

print(f"Total candidate pool in inverted index: {len(synth_cands):,} records.", flush=True)

# Build inverted index
inv = build_inverted_index(synth_cands)

blocked = 0
total_targets = 0
unblocked = []
total_cands_queried = 0

s1_dict = {sid: make_record(sid, name, addr, 'France') for sid, name, addr in fr_s1_records}

for sid, targets in synth_gt.items():
    r1 = s1_dict[sid]
    cands = query_candidates(r1, inv)
    total_cands_queried += len(cands)
    for t in targets:
        total_targets += 1
        if t in cands:
            blocked += 1
        else:
            unblocked.append((r1, synth_cands[t]))

recall = blocked / total_targets * 100.0
print(f"\n==================================================")
print(f"Scaled French Blocking Recall: {blocked}/{total_targets} = {recall:.2f}% (Missed: {len(unblocked)})")
print(f"Average candidates queried per S1 entity: {total_cands_queried / len(fr_s1_records):.1f}")
print(f"==================================================\n")

if unblocked:
    print("--- SAMPLE OF UNBLOCKED FRENCH PAIRS ---")
    for r1, r2 in unblocked[:5]:
        print(f"S1: {r1['norm_name']} | Addr: {r1['norm_address']}")
        print(f"C : {r2['norm_name']} | Addr: {r2['norm_address']}")
        print("-" * 50)
