import os
import re
import difflib
import warnings
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from business_entity_resolution import (
    normalize_text, extract_significant_tokens, strip_legal_suffixes,
    generate_acronyms, jaccard_similarity, levenshtein_ratio, token_sort_ratio,
    char_ngram_jaccard, compute_macro_f05, collective_resolve
)

# -------------------------------------------------------------
# 1. Parsers: Old vs New
# -------------------------------------------------------------
def old_parse_address(addr):
    if not isinstance(addr, str) or not addr.strip():
        return {'number': '', 'postcode': '', 'street': '', 'locality': ''}
    addr = addr.strip()
    p_match = re.search(r'\b\d{5,6}\b', addr)
    postcode = p_match.group(0) if p_match else ''
    clean_addr = re.sub(r'\b\d{5,6}\b', '', addr).strip()
    # Old parser ONLY checks leading numbers
    n_match = re.match(r'^(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', clean_addr, re.IGNORECASE)
    number = n_match.group(1).strip().lower() if n_match else ''
    if number:
        clean_addr = clean_addr[len(n_match.group(0)):].strip(', ')
    parts = [p.strip() for p in clean_addr.split(',') if p.strip()]
    locality = parts[-1].lower() if len(parts) > 1 else ''
    street = ', '.join(parts[:-1]).lower() if len(parts) > 1 else (parts[0].lower() if parts else '')
    return {'number': number, 'postcode': postcode, 'street': street, 'locality': locality}

def new_parse_address(addr):
    if not isinstance(addr, str) or not addr.strip():
        return {'number': '', 'postcode': '', 'street': '', 'locality': ''}
    addr = addr.strip()
    p_match = re.search(r'\b\d{5,6}\b', addr)
    postcode = p_match.group(0) if p_match else ''
    clean_addr = re.sub(r'\b\d{5,6}\b', '', addr).strip()
    number = ''
    pref_match = re.search(r'\b(?:no\.?|plot|door|bldg)\s*[:#-]?\s*(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', clean_addr, re.IGNORECASE)
    if pref_match:
        number = pref_match.group(1).strip().lower()
        clean_addr = clean_addr[:pref_match.start()] + clean_addr[pref_match.end():]
    else:
        lead_match = re.match(r'^(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', clean_addr, re.IGNORECASE)
        if lead_match:
            number = lead_match.group(1).strip().lower()
            clean_addr = clean_addr[lead_match.end():].strip(', ')
        else:
            trail_match = re.search(r'\b(?:rue|road|street|st|ave|avenue|bd|boulevard|lane)\s+[^,]*?\b(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', clean_addr, re.IGNORECASE)
            if trail_match:
                number = trail_match.group(1).strip().lower()
    parts = [p.strip() for p in clean_addr.split(',') if p.strip()]
    locality = parts[-1].lower() if len(parts) > 1 else ''
    street = ', '.join(parts[:-1]).lower() if len(parts) > 1 else (parts[0].lower() if parts else '')
    return {'number': number, 'postcode': postcode, 'street': street, 'locality': locality}

# -------------------------------------------------------------
# 2. Generators: Old (Locality Preserved) vs New (Broken Locality)
# -------------------------------------------------------------
def make_data(is_new_generator=True, n_entities=1200):
    np.random.seed(42)
    import random
    random.seed(42)

    brands = [
        "Starbucks Coffee", "Amazon Web Services", "Microsoft Technologies",
        "Google Cloud Platform", "Reliance Retail", "Tata Consultancy Services",
        "Infosys Technologies", "Apple Operations", "Nike Retail Stores",
        "Walmart Supercenter", "Costco Wholesale", "Target Brands",
        "IKEA Home Furnishings", "McDonalds Restaurants", "Subway Sandwiches"
    ]
    acronym_map = {
        "tata consultancy services": "tcs", "amazon web services": "aws",
        "reliance retail": "reliance", "mcdonalds restaurants": "mcdonalds"
    }
    cities = {
        'US': ['Seattle, WA', 'Redmond, WA', 'New York, NY', 'Austin, TX', 'San Francisco, CA'],
        'India': ['Bangalore, Karnataka', 'Gurgaon, Haryana', 'Mumbai, Maharashtra', 'Hyderabad, Telangana'],
        'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Bordeaux']
    }
    city_variants = {
        'Bangalore, Karnataka': ['Bengaluru', 'BLR', 'Bangalore'],
        'Gurgaon, Haryana': ['Gurugram', 'Gurgaon', 'GGN'],
        'Mumbai, Maharashtra': ['Bombay', 'Mumbai'],
        'New York, NY': ['NYC', 'New York', 'Manhattan'],
        'Paris': ['Paris 8e', 'Paris', 'Paris Cedex', 'Grand Paris']
    }
    landmarks = {
        'US': ['Near Central Station', 'Opposite City Center Mall', 'Airport Terminal 2', 'Main Street Plaza'],
        'India': ['Near SBI ATM', 'Opposite Metro Pillar 142', 'Behind Forum Mall', 'Railway Station Road'],
        'France': ['Pres de la Gare Centrale', 'En face du Metro', 'Pres de la Mairie', 'Avenue Victor Hugo']
    }

    s1_records, s2_records, s3_records = [], [], []
    ground_truth = {}

    def corrupt_name(name):
        lower = name.lower()
        for k, v in acronym_map.items():
            if k in lower and random.random() < 0.45:
                return name.lower().replace(k, v.upper())
        chars = list(name)
        if len(chars) > 6 and random.random() < 0.5:
            idx = random.randint(2, len(chars) - 2)
            chars.pop(idx)
        words = ''.join(chars).split()
        if len(words) > 2 and random.random() < 0.35:
            words[0], words[1] = words[1], words[0]
        return ' '.join(words)

    for i in range(n_entities):
        s1_id = f"S1-{i:05d}"
        base_brand = brands[i % len(brands)]
        brand_name = f"{base_brand} {i//len(brands) + 1}"
        country = 'France' if i >= int(n_entities * 0.75) else ('US' if i % 2 == 0 else 'India')
        base_city = cities[country][(i * 3) % len(cities[country])]
        num_val = 10 + (i % 500)

        if is_new_generator:
            if country == 'France':
                s1_addr = f"Rue de la Paix {num_val}, {base_city}"
            elif country == 'India':
                s1_addr = f"Plot No. {num_val}, MG Road, {base_city}"
            else:
                s1_addr = f"{num_val} Main St, {base_city}"
        else:
            s1_addr = f"{num_val} Main St, {base_city}"

        s1_records.append({
            'entity_id': s1_id,
            'business_name': f"{brand_name} Ltd" if country == 'India' else f"{brand_name} Inc",
            'business_address': s1_addr,
            'country': country
        })

        match_type = np.random.choice(['singleton', 'single', 'multi'], p=[0.25, 0.35, 0.40])
        matches = []

        if match_type == 'single':
            s2_id = f"S2-{len(s2_records):05d}"
            noisy_name = corrupt_name(brand_name)
            if is_new_generator:
                loc_noise = "" if random.random() < 0.25 else city_variants.get(base_city, [base_city])[0]
                if random.random() < 0.5:
                    noisy_addr = f"{random.choice(landmarks[country])}, {loc_noise}" if loc_noise else random.choice(landmarks[country])
                else:
                    noisy_addr = s1_addr.replace("Main St", "Main Street").replace("MG Road", "M.G. Rd")
                    if loc_noise:
                        noisy_addr = noisy_addr.replace(base_city, loc_noise)
            else:
                if random.random() < 0.5:
                    noisy_addr = f"{random.choice(landmarks[country])}, {base_city}"
                else:
                    noisy_addr = s1_addr.replace("Main St", "Main Street").replace("Bangalore", "Bengaluru")

            s2_records.append({
                'entity_id': s2_id,
                'business_name': noisy_name,
                'business_address': noisy_addr,
                'country': country
            })
            matches.append(s2_id)

        elif match_type == 'multi':
            s2_id = f"S2-{len(s2_records):05d}"
            s3_id = f"S3-{len(s3_records):05d}"
            s2_records.append({
                'entity_id': s2_id,
                'business_name': corrupt_name(brand_name),
                'business_address': s1_addr.replace("Main St", "M. St"),
                'country': country
            })
            s3_records.append({
                'entity_id': s3_id,
                'business_name': brand_name.split()[0],
                'business_address': f"{random.choice(landmarks[country])}, {base_city}",
                'country': country
            })
            matches.extend([s2_id, s3_id])

        ground_truth[s1_id] = matches

    # Hard Negatives
    for k in range(350):
        country = 'France' if k >= 250 else ('US' if k % 2 == 0 else 'India')
        base_city = cities[country][k % len(cities[country])]
        brand = brands[k % len(brands)] + f" {k % 30 + 1}"
        s2_records.append({
            'entity_id': f"S2-BRANCH-{k:04d}",
            'business_name': f"{brand} Airport Branch",
            'business_address': f"Airport Terminal {1 + (k%3)}, {base_city}",
            'country': country
        })
        s3_records.append({
            'entity_id': f"S3-CONFUSE-{k:04d}",
            'business_name': f"{brand.split()[0]} Logistics Group",
            'business_address': f"{50 + k} Commercial Complex, {base_city}",
            'country': country
        })

    return pd.DataFrame(s1_records), pd.DataFrame(s2_records), pd.DataFrame(s3_records), ground_truth

# -------------------------------------------------------------
# 3. Pipeline Runner for Specific Ablation Configuration
# -------------------------------------------------------------
def run_ablation(is_new_generator, is_new_parser, threshold):
    df_s1, df_s2, df_s3, ground_truth = make_data(is_new_generator=is_new_generator)
    df_cands = pd.concat([df_s2, df_s3], ignore_index=True)

    parser_fn = new_parse_address if is_new_parser else old_parse_address

    for df in [df_s1, df_cands]:
        df['norm_name'] = df['business_name'].apply(lambda x: normalize_text(x, is_address=False))
        df['norm_address'] = df['business_address'].apply(lambda x: normalize_text(x, is_address=True))
        df['sig_tokens'] = df['norm_name'].apply(extract_significant_tokens)
        df['acronyms'] = df['norm_name'].apply(generate_acronyms)
        df['parsed_addr'] = df['business_address'].apply(parser_fn)
        df['stripped_name'] = df['norm_name'].apply(strip_legal_suffixes)

    all_names = list(df_s1['norm_name']) + list(df_cands['norm_name'])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(all_names)
    s1_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_s1.iterrows()}
    cand_tfidf_map = {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_candidates.iterrows()} if 'df_candidates' in locals() else {row['entity_id']: tfidf.transform([row['norm_name']]) for _, row in df_cands.iterrows()}

    from business_entity_resolution import build_blocking_candidates, extract_pairwise_features
    pairs, _ = build_blocking_candidates(df_s1, df_cands)

    df_s1_dict = df_s1.set_index('entity_id').to_dict('index')
    df_cand_dict = df_cands.set_index('entity_id').to_dict('index')
    X = extract_pairwise_features(pairs, df_s1_dict, df_cand_dict, s1_tfidf_map, cand_tfidf_map)

    flat_gt = set((sid, m) for sid, ml in ground_truth.items() for m in ml)
    y = np.array([1 if p in flat_gt else 0 for p in pairs])

    france_s1 = set(df_s1[df_s1['country'] == 'France']['entity_id'])
    non_france = df_s1[df_s1['country'] != 'France']['entity_id'].values
    np.random.seed(42); np.random.shuffle(non_france)
    split = int(len(non_france)*0.75)
    train_ids = set(non_france[:split])
    test_ids = france_s1

    s1_idx = X.index.get_level_values('s1')
    X_train, y_train = X[s1_idx.isin(train_ids)], y[s1_idx.isin(train_ids)]
    X_test = X[s1_idx.isin(test_ids)]

    clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)

    test_probs = clf.predict_proba(X_test)[:, 1]
    test_gt = {sid: ground_truth[sid] for sid in test_ids}
    coll_test = collective_resolve(X_test.index, test_probs, test_ids, primary_t=threshold, secondary_t=threshold+0.15, max_margin=0.25)
    f05, p, r = compute_macro_f05(test_gt, coll_test)

    # Scrutinize accepted matches: find minimum probability and bottom accepted pairs
    accepted_info = []
    test_pairs_list = list(X_test.index)
    for sid, m_list in coll_test.items():
        for cid in m_list:
            idx = test_pairs_list.index((sid, cid))
            accepted_info.append((sid, cid, test_probs[idx], cid in test_gt[sid]))

    accepted_info.sort(key=lambda x: x[2])
    min_p = accepted_info[0][2] if accepted_info else 0.0

    return f05, p, r, min_p, accepted_info, df_s1_dict, df_cand_dict

def main():
    print("=" * 75)
    print("  AGENT.MD ABLATION AUDIT: ISOLATING PARSER VS GENERATOR VS THRESHOLD  ")
    print("=" * 75)

    experiments = [
        ("Exp 1: Old Gen (Locality Preserved) + Old Parser + Thresh 0.20", False, False, 0.20),
        ("Exp 2: Old Gen (Locality Preserved) + NEW Parser + Thresh 0.20", False, True, 0.20),
        ("Exp 3: NEW Gen (Broken Locality)   + NEW Parser + Thresh 0.20", True, True, 0.20),
        ("Exp 4: NEW Gen (Broken Locality)   + NEW Parser + Thresh 0.30", True, True, 0.30),
    ]

    results = []
    last_accepted = None
    for name, is_new_g, is_new_p, t in experiments:
        f05, p, r, min_p, acc_info, s1_d, c_d = run_ablation(is_new_g, is_new_p, t)
        results.append((name, f05, p, r, min_p, len(acc_info)))
        last_accepted = (acc_info, s1_d, c_d)

    print("\n  Config | Macro F0.5 | Precision | Recall | Min Accepted Prob | Total Accepted")
    print("  " + "-" * 80)
    for name, f05, p, r, min_p, count in results:
        print(f"  {name[:48]:<48} |  {f05:.4f}  |  {p*100:5.1f}%  | {r*100:5.1f}% |      {min_p:.3f}       |    {count}")

    # Inspect the absolute closest accepted calls
    acc_info, s1_d, c_d = last_accepted
    print("\n" + "=" * 75)
    print("  EYEBALLING THE ABSOLUTE LOWEST-PROBABILITY ACCEPTED MATCHES (CLOSEST CALLS)  ")
    print("=" * 75)
    for sid, cid, prob, is_true in acc_info[:5]:
        r1, r2 = s1_d[sid], c_d[cid]
        verdict = "TRUE MATCH (Correct)" if is_true else "FALSE MERGE (Error!)"
        print(f"  • S1: [{r1['business_name']}] @ [{r1['business_address']}]")
        print(f"    Candidate: [{r2['business_name']}] @ [{r2['business_address']}]")
        print(f"    Accepted Probability: {prob:.3f} | Ground-Truth: {verdict}\n")

if __name__ == '__main__':
    main()
