import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from business_entity_resolution import (
    normalize_text, extract_significant_tokens, generate_acronyms,
    parse_address_components, strip_legal_suffixes,
    extract_pairwise_features
)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

s1 = {
    'entity_id': 'S1-TEST',
    'business_name': 'Nike Retail Stores 79 Inc',
    'business_address': 'Rue de la Paix 188, Bordeaux',
    'country': 'France'
}

cand = {
    'entity_id': 'S2-TEST',
    'business_name': 'Retail Nike Stores 79',
    'business_address': 'R. de la Paix 188',
    'country': 'France'
}

df_s1 = pd.DataFrame([s1])
df_cand = pd.DataFrame([cand])

for df in [df_s1, df_cand]:
    df['norm_name'] = df['business_name'].apply(lambda x: normalize_text(x, is_address=False))
    df['norm_address'] = df['business_address'].apply(lambda x: normalize_text(x, is_address=True))
    df['sig_tokens'] = df['norm_name'].apply(extract_significant_tokens)
    df['acronyms'] = df['norm_name'].apply(generate_acronyms)
    df['parsed_addr'] = df['business_address'].apply(parse_address_components)
    df['stripped_name'] = df['norm_name'].apply(strip_legal_suffixes)

all_names = list(df_s1['norm_name']) + list(df_cand['norm_name'])
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(all_names)
s1_tfidf = {'S1-TEST': tfidf.transform([df_s1.iloc[0]['norm_name']])}
cand_tfidf = {'S2-TEST': tfidf.transform([df_cand.iloc[0]['norm_name']])}

pairs = [('S1-TEST', 'S2-TEST')]
s1_dict = df_s1.set_index('entity_id').to_dict('index')
cand_dict = df_cand.set_index('entity_id').to_dict('index')

X = extract_pairwise_features(pairs, s1_dict, cand_dict, s1_tfidf, cand_tfidf)

print("Parsed address S1:  ", s1_dict['S1-TEST']['parsed_addr'])
print("Parsed address Cand:", cand_dict['S2-TEST']['parsed_addr'])
print("\nExtracted features:")
for col in X.columns:
    print(f"  {col:<25}: {X.iloc[0][col]}")
