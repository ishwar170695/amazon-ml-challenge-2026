import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import unicodedata
import re

def has_indic(text):
    if not isinstance(text, str):
        return False
    for ch in text:
        name = unicodedata.name(ch, '')
        if any(s in name for s in ['DEVANAGARI', 'TAMIL', 'TELUGU', 'BENGALI', 'GUJARATI', 'GURMUKHI', 'KANNADA', 'MALAYALAM']):
            return True
    return False

# Load 1,000 S1 records from India
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s1_india = s1_df[s1_df['country'] == 'India'].head(500)
s1_ids = set(s1_india['entity_id'])

# Load GT
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_india = gt_df[gt_df['source1_entity_id'].isin(s1_ids)]
gt_map = {}
all_matched_cands = set()
for _, r in gt_india.iterrows():
    m = [x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip() and x.strip() != 'nan']
    gt_map[r['source1_entity_id']] = m
    all_matched_cands.update(m)

print(f"Sampled {len(s1_ids)} S1 India entities with {len(all_matched_cands)} true matches.")

# Load matching S2/S3 records
cands_data = {}
for split_name in ['train_source2.tsv', 'train_source3.tsv']:
    for chunk in pd.read_csv(f'dataset/train/{split_name}', sep='\t', chunksize=100000):
        m = chunk[chunk['entity_id'].isin(all_matched_cands)]
        for _, r in m.iterrows():
            cands_data[r['entity_id']] = r.to_dict()
        if len(cands_data) >= len(all_matched_cands):
            break

print(f"Found {len(cands_data)} / {len(all_matched_cands)} candidate records in early chunks.")

# Test blocking recall with:
# A. Name tokens only (current)
# B. Name tokens + Address distinctive tokens/numbers (proposed)

def get_tokens(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower()))

def get_addr_numbers(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d{3,6}\b', text))

hits_name_only = 0
hits_with_addr = 0
total_pairs = 0
indic_pairs = 0
indic_hits_name_only = 0
indic_hits_with_addr = 0

for s1_id, match_list in gt_map.items():
    s1_row = s1_india[s1_india['entity_id'] == s1_id].iloc[0]
    s1_name_toks = get_tokens(s1_row['business_name'])
    s1_addr_toks = get_tokens(s1_row['business_address'])
    s1_addr_nums = get_addr_numbers(s1_row['business_address'])

    for cid in match_list:
        if cid not in cands_data:
            continue
        c_row = cands_data[cid]
        c_name_toks = get_tokens(c_row['business_name'])
        c_addr_toks = get_tokens(c_row['business_address'])
        c_addr_nums = get_addr_numbers(c_row['business_address'])

        total_pairs += 1
        is_indic = has_indic(c_row['business_name'])
        if is_indic:
            indic_pairs += 1

        # Check Name-only overlap
        name_hit = bool(s1_name_toks & c_name_toks)
        if name_hit:
            hits_name_only += 1
            if is_indic:
                indic_hits_name_only += 1

        # Check Name OR Address overlap
        # distinct addr match: shared number OR >= 2 shared addr tokens
        addr_hit = bool(s1_addr_nums & c_addr_nums) or (len(s1_addr_toks & c_addr_toks) >= 2)
        combined_hit = name_hit or addr_hit
        if combined_hit:
            hits_with_addr += 1
            if is_indic:
                indic_hits_with_addr += 1

print("\n" + "="*50)
print(f"Total True Pairs Evaluated: {total_pairs}")
print(f"Indic Name Pairs: {indic_pairs} ({indic_pairs/total_pairs*100:.1f}%)")
print(f"Recall (Name-only blocking):        {hits_name_only}/{total_pairs} ({hits_name_only/total_pairs*100:.2f}%)")
print(f"Recall (Name + Address blocking):   {hits_with_addr}/{total_pairs} ({hits_with_addr/total_pairs*100:.2f}%)")
if indic_pairs > 0:
    print(f"Indic Recall (Name-only):         {indic_hits_name_only}/{indic_pairs} ({indic_hits_name_only/indic_pairs*100:.2f}%)")
    print(f"Indic Recall (Name + Address):    {indic_hits_with_addr}/{indic_pairs} ({indic_hits_with_addr/indic_pairs*100:.2f}%)")
print("="*50)
