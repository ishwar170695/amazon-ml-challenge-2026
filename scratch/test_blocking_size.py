import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re
from collections import Counter

# Load 1,000 S1 and 5,000 S2/S3
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=1000)
s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=5000)
s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=5000)
cands_df = pd.concat([s2_df, s3_df], ignore_index=True)

def get_tokens(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower()))

def get_addr_numbers(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d{3,6}\b', text))

# Build token frequency to filter ultra-frequent stop words
name_token_counts = Counter()
for name in cands_df['business_name']:
    name_token_counts.update(get_tokens(name))

# Inverted index
inv_name = {}
inv_num = {}

for idx, r in cands_df.iterrows():
    cid = r['entity_id']
    c_country = r['country']
    # Name tokens (filter words appearing in > 2% of pool)
    for tok in get_tokens(r['business_name']):
        if name_token_counts[tok] < len(cands_df) * 0.02:
            inv_name.setdefault((c_country, tok), []).append(cid)
    # Address numbers
    for num in get_addr_numbers(r['business_address']):
        inv_num.setdefault((c_country, num), []).append(cid)

cands_per_s1 = []
for idx, r in s1_df.iterrows():
    country = r['country']
    candidates = set()
    for tok in get_tokens(r['business_name']):
        for cid in inv_name.get((country, tok), []):
            candidates.add(cid)
    for num in get_addr_numbers(r['business_address']):
        for cid in inv_num.get((country, num), []):
            candidates.add(cid)
    cands_per_s1.append(len(candidates))

s = pd.Series(cands_per_s1)
print("=== CANDIDATE POOL SIZE PER S1 ENTITY ===")
print(s.describe())
print(f"Total candidate pairs generated: {s.sum():,} out of {len(s1_df) * len(cands_df):,} total comparisons")
print(f"Reduction Ratio: {100 * (1 - s.sum() / (len(s1_df) * len(cands_df))):.4f}%")
