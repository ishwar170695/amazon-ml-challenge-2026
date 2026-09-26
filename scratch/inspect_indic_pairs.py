import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import unicodedata

def has_indic(text):
    if not isinstance(text, str):
        return False
    for ch in text:
        name = unicodedata.name(ch, '')
        if any(script in name for script in ['DEVANAGARI', 'TAMIL', 'TELUGU', 'BENGALI', 'GUJARATI', 'GURMUKHI', 'KANNADA', 'MALAYALAM', 'ORIYA']):
            return True
    return False

# Read GT
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10000).set_index('entity_id')
s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=50000).set_index('entity_id')

count = 0
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_df.index:
        continue
    s1_row = s1_df.loc[s1_id]
    if s1_row['country'] != 'India':
        continue
    cands = [x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip()]
    for cid in cands:
        if cid in s2_df.index:
            s2_row = s2_df.loc[cid]
            if has_indic(s2_row['business_name']) or has_indic(s2_row['business_address']):
                print('='*60)
                print(f"S1: [{s1_id}]")
                print(f"  Name: {s1_row['business_name']}")
                print(f"  Addr: {s1_row['business_address']}")
                print(f"MATCH S2: [{cid}]")
                print(f"  Name: {s2_row['business_name']}")
                print(f"  Addr: {s2_row['business_address']}")
                count += 1
                if count >= 5:
                    break
    if count >= 5:
        break
