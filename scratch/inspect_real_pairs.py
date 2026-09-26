import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# Load ground truth sample
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=50)
gt_matches = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')].head(5)

s1_ids = set(gt_matches['source1_entity_id'])
all_s2_s3_ids = set()
for m in gt_matches['matched_entity_ids']:
    all_s2_s3_ids.update(m.split(','))

# Load S1
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s1_sample = s1_df[s1_df['entity_id'].isin(s1_ids)].set_index('entity_id')

s2_sample = {}
s3_sample = {}

for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=200000):
    matched = chunk[chunk['entity_id'].isin(all_s2_s3_ids)]
    for _, row in matched.iterrows():
        s2_sample[row['entity_id']] = row.to_dict()
    if len([x for x in all_s2_s3_ids if x.startswith('S2-')]) > 0 and len(s2_sample) >= len([x for x in all_s2_s3_ids if x.startswith('S2-')]):
        break

for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=200000):
    matched = chunk[chunk['entity_id'].isin(all_s2_s3_ids)]
    for _, row in matched.iterrows():
        s3_sample[row['entity_id']] = row.to_dict()
    if len([x for x in all_s2_s3_ids if x.startswith('S3-')]) > 0 and len(s3_sample) >= len([x for x in all_s2_s3_ids if x.startswith('S3-')]):
        break

for _, row in gt_matches.iterrows():
    s1_id = row['source1_entity_id']
    target_ids = row['matched_entity_ids'].split(',')
    s1_row = s1_sample.loc[s1_id]
    c = s1_row['country']
    print('='*70)
    print(f'S1: [{s1_id}] ({c})')
    print(f'    Name   : {s1_row["business_name"]}')
    print(f'    Address: {s1_row["business_address"]}')
    print('  MATCHES:')
    for tid in target_ids:
        trow = s2_sample.get(tid) or s3_sample.get(tid)
        if trow:
            print(f'    [{tid}] ({trow["country"]})')
            print(f'      Name   : {trow["business_name"]}')
            print(f'      Address: {trow["business_address"]}')
        else:
            print(f'    [{tid}] (not found in early chunks)')
