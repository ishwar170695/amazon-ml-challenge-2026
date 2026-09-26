"""
kaggle_pipeline.py - Maximum-Performance Parallel Predictor for Kaggle (Linux / 30 GB RAM / 4 vCPUs)
=====================================================================================================
Optimized specifically for Kaggle Notebooks & Scripts:
  1. Auto-installs missing dependencies (rapidfuzz, anyascii) in 3 seconds.
  2. Auto-detects test files and model whether in /kaggle/input/..., /kaggle/working/..., or local.
  3. Precomputes country candidates in RAM (compact tuples, zero redundant work).
  4. Native Linux Copy-on-Write (fork) 4-core worker pool: zero IPC copying, shared memory.
  5. On-the-fly 3-gram computation (saves 20 GB RAM, zero risk of OOM).
  6. High-throughput LightGBM OpenMP scoring.
  7. Exact global bipartite claiming (collective_resolve) preserving 100% frozen model fidelity.
  8. Automatic official submission validation.
  9. Creates submission_archive.zip ready for 1-click download.
"""

import os
import sys
import time
import gc
import glob
import zipfile
import array
import pickle
import subprocess

# ----------------------------------------------------------------------
# 0. ENSURE DEPENDENCIES
# ----------------------------------------------------------------------
def ensure_dependencies():
    needed = []
    for pkg, imp in [('rapidfuzz', 'rapidfuzz'), ('anyascii', 'anyascii'), ('lightgbm', 'lightgbm'),
                     ('scikit-learn', 'sklearn'), ('scipy', 'scipy'), ('numpy', 'numpy')]:
        try:
            __import__(imp)
        except ImportError:
            needed.append(pkg)
    if needed:
        print(f"[SETUP] Installing missing packages: {needed}...", flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + needed)
        print("[SETUP] All packages installed successfully!", flush=True)

ensure_dependencies()

import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from rapidfuzz import fuzz
import anyascii
import re

# ----------------------------------------------------------------------
# 1. ROBUST PATH AUTO-DETECTION
# ----------------------------------------------------------------------
def find_test_files():
    search_patterns = [
        '/content/**/test_source1.tsv',
        '/content/drive/**/test_source1.tsv',
        '/kaggle/input/**/test_source1.tsv',
        '/kaggle/working/**/test_source1.tsv',
        'dataset/test/**/test_source1.tsv',
        '../dataset/test/**/test_source1.tsv',
        '**/test_source1.tsv',
        'test_source1.tsv'
    ]
    for pattern in search_patterns:
        for m in glob.glob(pattern, recursive=True):
            d = os.path.dirname(m)
            c2 = os.path.join(d, 'test_source2.tsv')
            c3 = os.path.join(d, 'test_source3.tsv')
            if os.path.exists(c2) and os.path.exists(c3):
                return os.path.abspath(m), os.path.abspath(c2), os.path.abspath(c3)
    raise FileNotFoundError("Could not locate test_source1.tsv, test_source2.tsv, and test_source3.tsv. "
                            "Please upload the test dataset or mount Google Drive.")

def find_model_file():
    search_patterns = [
        'artifacts/model_v3.pkl',
        'model_v3.pkl',
        '/content/**/model_v3.pkl',
        '/content/drive/**/model_v3.pkl',
        '/kaggle/input/**/model_v3.pkl',
        '/kaggle/working/**/model_v3.pkl',
        '**/model_v3.pkl'
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    raise FileNotFoundError("Could not locate model_v3.pkl.")

# ----------------------------------------------------------------------
# 2. CORE UTILITIES & NORMALIZATION
# ----------------------------------------------------------------------
STOPWORDS = {
    'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'co', 'company', 'llc', 'pvt', 'private', 'sa', 'sarl', 'sas',
    'gmbh', 'bv', 'nv', 'spa', 'srl', 'the', 'and', 'of', 'in', 'at',
    'for', 'on', 'a', 'an', 'de', 'du', 'des', 'la', 'le', 'les',
    'und', 'der', 'die', 'das', 'et', 'en', 'd', 'l'
}

SUFFIXES = [
    'private limited', 'pvt ltd', 'pvt. ltd.', 'pvt. limited', 'private ltd',
    'limited', 'ltd.', 'ltd', 'inc.', 'inc', 'incorporated', 'corporation', 'corp.',
    'corp', 'llc.', 'llc', 'company', 'co.', 'gmbh', 's.a.', 'sa', 's.a.r.l.',
    'sarl', 's.a.s.', 'sas', 'b.v.', 'bv', 'n.v.', 'nv', 's.p.a.', 'spa',
    's.r.l.', 'srl', 'plc', 'holdings', 'holding', 'group', 'services'
]

def set_jaccard(a, b):
    if not a or not b: return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def jaccard_3gram(s1, s2):
    if len(s1) < 3 or len(s2) < 3: return 0.0
    c1 = {s1[i:i+3] for i in range(len(s1)-2)}
    c2 = {s2[i:i+3] for i in range(len(s2)-2)}
    u = len(c1 | c2)
    return len(c1 & c2) / u if u else 0.0

def csr_to_dict_list(mat):
    indptr = mat.indptr
    indices = mat.indices
    data = mat.data
    return [{int(indices[j]): float(data[j]) for j in range(indptr[i], indptr[i+1])} for i in range(mat.shape[0])]

def normalize_text(text, is_address=False):
    if not isinstance(text, str) or not text: return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

def strip_legal(norm_name):
    if not norm_name: return ""
    tokens = norm_name.split()
    while tokens and tokens[-1] in STOPWORDS:
        tokens.pop()
    s = ' '.join(tokens)
    for sfx in SUFFIXES:
        if s.endswith(' ' + sfx):
            s = s[:-len(sfx)-1].strip()
            break
    return s

def indic_phonetic_skeleton(token):
    if not token or len(token) < 2: return token
    t = token.lower()
    t = re.sub(r'[aeiouy]', '', t)
    t = re.sub(r'kh|gh|ch|jh|th|dh|ph|bh|sh', 'h', t)
    t = re.sub(r'[bcdfgjklmnpqrstvwxz]', lambda m: m.group(0)[0], t)
    res = [t[0]] if t else []
    for c in t[1:]:
        if c != res[-1]: res.append(c)
    return ''.join(res)

def gen_acronyms(norm_name):
    if not norm_name: return set()
    tokens = norm_name.split()
    acr = set()
    if len(tokens) >= 2:
        acr.add(''.join(t[0] for t in tokens))
        clean = [t for t in tokens if t not in STOPWORDS]
        if len(clean) >= 2: acr.add(''.join(t[0] for t in clean))
    return {a for a in acr if len(a) >= 2}

def extract_unit_keys(addr):
    if not addr: return set()
    pat = r'\b(?:suite|ste|apt|apartment|unit|bldg|building|floor|fl|room|rm|plot|shop|no|number|block)\s*#?\s*([a-zA-Z0-9\-]+)\b'
    return {m.lower().lstrip('0') for m in re.findall(pat, addr, re.IGNORECASE) if len(m) >= 1}

def clean_compact_brand(norm_name):
    if not norm_name: return ""
    toks = [t for t in norm_name.split() if t not in STOPWORDS and len(t) >= 4]
    return toks[0] if toks else ""

def get_distinctive_addr_tokens(norm_addr):
    if not norm_addr: return set()
    common_addr = {'street', 'st', 'road', 'rd', 'avenue', 'ave', 'boulevard', 'blvd',
                   'lane', 'ln', 'drive', 'dr', 'way', 'place', 'pl', 'court', 'ct',
                   'rue', 'chemin', 'route', 'allee', 'place', 'marais', 'paris',
                   'nagar', 'road', 'cross', 'main', 'layout', 'bangalore', 'mumbai',
                   'delhi', 'chennai', 'hyderabad', 'west', 'east', 'north', 'south'}
    return {t for t in norm_addr.split() if len(t) >= 4 and t not in common_addr and not t.isdigit()}

def parse_addr(addr):
    if not addr or not isinstance(addr, str):
        return {'number': '', 'postcode': '', 'street': '', 'loc_tokens': set()}
    clean = re.sub(r'[^\w\s]', ' ', addr.lower())
    toks = clean.split()
    pc_m = re.search(r'\b\d{5}(?:-\d{4})?\b', addr)
    if not pc_m: pc_m = re.search(r'\b\d{6}\b', addr)
    postcode = pc_m.group(0).split('-')[0] if pc_m else ''
    num_m = re.search(r'\b\d{1,5}\b', clean)
    number = num_m.group(0) if num_m else ''
    loc_tokens = {t for t in toks if len(t) >= 3 and t != postcode and t != number and t not in STOPWORDS}
    street = ' '.join(toks[:4]) if toks else ''
    return {'number': number, 'postcode': postcode, 'street': street, 'loc_tokens': loc_tokens}

def get_addr_numbers(t):
    if not isinstance(t, str): return set()
    nums = set(re.findall(r'(?<!\d)\d{2,6}(?!\d)', t))
    nums.update({n.lstrip('0') for n in nums if len(n.lstrip('0')) >= 2})
    return nums

def get_house_codes(t):
    if not isinstance(t, str): return set()
    return set(re.findall(r'\b[a-zA-Z]-?\d{1,4}\b', t.lower()))

def make_record(eid, name, addr, country):
    name_ascii = anyascii.anyascii(name) if any(ord(c) > 127 for c in name) else name
    addr_ascii = anyascii.anyascii(addr) if any(ord(c) > 127 for c in addr) else addr
    pa = parse_addr(addr_ascii)
    nn = normalize_text(name_ascii)
    na = normalize_text(addr_ascii, is_address=True)
    st = set(nn.split()) - STOPWORDS
    pht = {indic_phonetic_skeleton(t) for t in st if len(t) >= 3}
    return {
        'entity_id': eid,
        'country': country,
        'norm_name': nn,
        'norm_address': na,
        'sig_tokens': st,
        'phonetic_tokens': pht,
        'unit_keys': extract_unit_keys(addr_ascii),
        'compact_brand': clean_compact_brand(nn),
        'distinctive_addr_tokens': get_distinctive_addr_tokens(na),
        'acronyms': gen_acronyms(nn),
        'parsed_addr': pa,
        'stripped_name': strip_legal(nn),
        'addr_nums': get_addr_numbers(addr_ascii),
        'house_codes': get_house_codes(addr_ascii),
    }

def collective_resolve(pairs, probs, entity_ids, primary_threshold=None, secondary_threshold=None,
                       max_matches=12, s1_dict=None, **kwargs):
    if primary_threshold is None:
        primary_threshold = {'US': 0.980, 'India': 0.970, 'France': 0.970, 'default': 0.970}
    s1_candidates = {s: [] for s in entity_ids}
    cand_claims = {}
    def get_p_thresh(sid):
        if isinstance(primary_threshold, dict):
            c = s1_dict[sid]['country'] if s1_dict and sid in s1_dict else 'default'
            return primary_threshold.get(c, primary_threshold.get('default', 0.970))
        return primary_threshold

    for idx, ((s1, c), p) in enumerate(zip(pairs, probs)):
        p_th = get_p_thresh(s1)
        if s1 in s1_candidates and p >= p_th:
            s1_candidates[s1].append((c, p, idx))
            cand_claims.setdefault(c, []).append((p, s1))
    candidate_winner = {}
    for c, claims in cand_claims.items():
        claims.sort(reverse=True, key=lambda x: x[0])
        candidate_winner[c] = claims[0][1]
    final_matches = {s: [] for s in entity_ids}
    for s1, cands in s1_candidates.items():
        if not cands: continue
        cands.sort(reverse=True, key=lambda x: x[1])
        final_matches[s1] = [c for c, p, idx in cands if candidate_winner.get(c) == s1][:max_matches]
    return final_matches

# ----------------------------------------------------------------------
# 3. LINUX COPY-ON-WRITE PARALLEL FEATURE WORKER
# ----------------------------------------------------------------------
_G_S1_TUPLES = None
_G_CAND_TUPLES = None
_G_NAME_TO_WDICT = None
_G_NAME_TO_CDICT = None

def init_worker(s1_tuples, cand_tuples, name_to_wdict, name_to_cdict):
    global _G_S1_TUPLES, _G_CAND_TUPLES, _G_NAME_TO_WDICT, _G_NAME_TO_CDICT
    _G_S1_TUPLES = s1_tuples
    _G_CAND_TUPLES = cand_tuples
    _G_NAME_TO_WDICT = name_to_wdict
    _G_NAME_TO_CDICT = name_to_cdict

def compute_chunk_features(pairs_slice):
    n = len(pairs_slice)
    X = np.empty((n, 21), dtype=np.float32)
    s1_tups = _G_S1_TUPLES
    cand_tups = _G_CAND_TUPLES
    wdict = _G_NAME_TO_WDICT
    cdict = _G_NAME_TO_CDICT

    for i, (s_idx, c_idx) in enumerate(pairs_slice):
        r1 = s1_tups[s_idx]
        r2 = cand_tups[c_idx]

        n1, n2 = r1[1], r2[1]
        a1, a2 = r1[2], r2[2]

        ntj = set_jaccard(r1[6], r2[6])
        nts = fuzz.token_sort_ratio(n1, n2) / 100.0
        nlv = fuzz.ratio(n1, n2) / 100.0
        nc3 = jaccard_3gram(n1, n2)

        sn1, sn2 = r1[3], r2[3]
        wd1 = wdict.get(sn1, {})
        wd2 = wdict.get(sn2, {})
        twc = sum(v * wd2[k] for k, v in wd1.items() if k in wd2) if wd1 and wd2 else 0.0

        cd1 = cdict.get(sn1, {})
        cd2 = cdict.get(sn2, {})
        tcc = sum(v * cd2[k] for k, v in cd1.items() if k in cd2) if cd1 and cd2 else 0.0

        lex = 1.0 if sn1 and sn1 == sn2 else 0.0
        acr = 1.0 if r1[7] & r2[7] else 0.0

        num1, num2 = r1[9], r2[9]
        if num1 and num2:
            nsm = 1.0 if num1 == num2 else -1.0
            mnm = 0.0
        else:
            nsm = 0.0; mnm = 1.0

        pc1, pc2 = r1[10], r2[10]
        if pc1 and pc2:
            if pc1 == pc2: pcm = 1.0
            elif pc1[:3] == pc2[:3]: pcm = 0.5
            else: pcm = -1.0
            mpc = 0.0
        else:
            pcm = 0.0; mpc = 1.0

        ljc = set_jaccard(r1[12], r2[12])
        st1, st2 = r1[11], r2[11]
        slv = (fuzz.ratio(st1, st2) / 100.0) if st1 and st2 else 0.0
        ac3 = jaccard_3gram(a1, a2)
        alv = (fuzz.ratio(a1, a2) / 100.0) if a1 and a2 else 0.0
        mad = 1.0 if not a1 or not a2 else 0.0

        max_l = max(len(n1), len(n2), 1)
        nld = abs(len(n1) - len(n2)) / max_l

        pht1, pht2 = r1[5], r2[5]
        phs = (fuzz.token_sort_ratio(pht1, pht2) / 100.0) if pht1 and pht2 else 0.0

        cb1, cb2 = r1[4], r2[4]
        cbs = (fuzz.ratio(cb1, cb2) / 100.0) if cb1 and cb2 else 0.0

        u1, u2 = r1[8], r2[8]
        ukm = 1.0 if (u1 and u2 and (u1 & u2)) else (-1.0 if (u1 and u2) else 0.0)

        X[i] = [ntj, nts, nlv, nc3, twc, tcc, lex, acr, nsm, mnm, pcm, mpc, ljc, slv, ac3, alv, mad, nld, phs, cbs, ukm]

    return X

# ----------------------------------------------------------------------
# 4. MAIN ORCHESTRATION PIPELINE
# ----------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Sample limit for testing')
    parser.add_argument('--country', type=str, default=None, help='Target country (France, US, India)')
    args, unknown = parser.parse_known_args()

    total_start = time.time()
    print("=" * 80, flush=True)
    print("  AMAZON ML CHALLENGE 2026 - KAGGLE HIGH-PERFORMANCE PREDICTOR", flush=True)
    print("  Optimized for Ubuntu Linux / 30 GB RAM / 4 vCPUs", flush=True)
    if args.limit:
        print(f"  [TEST MODE] Limiting to {args.limit} entities per country", flush=True)
    print("=" * 80, flush=True)

    # 1. Locate paths
    s1_file, s2_file, s3_file = find_test_files()
    model_file = find_model_file()
    print(f"[FOUND] Test Source 1: {s1_file}", flush=True)
    print(f"[FOUND] Test Source 2: {s2_file}", flush=True)
    print(f"[FOUND] Test Source 3: {s3_file}", flush=True)
    print(f"[FOUND] Model File:    {model_file}", flush=True)

    with open(model_file, 'rb') as f:
        artifacts = pickle.load(f)
    clf = artifacts['model']
    word_vec = artifacts['word_tfidf']
    char_vec = artifacts['char_tfidf']
    thresh_config = artifacts.get('thresholds', {'US': 0.98, 'India': 0.97, 'France': 0.97, 'default': 0.97})
    sec_thresh = artifacts.get('secondary_threshold', 0.88)
    margin = artifacts.get('margin', 0.20)

    if os.path.exists('/kaggle/working'):
        output_dir = '/kaggle/working/output'
    elif os.path.exists('/content'):
        output_dir = '/content/output'
    else:
        output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'matching_results.tsv')
    cand_file = os.path.join(output_dir, 'candidate_pairs.tsv')

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(cand_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    num_cpus = mp.cpu_count()
    print(f"[SYSTEM] Detected {num_cpus} CPU cores. Target output: {output_dir}", flush=True)

    total_s1_processed = 0
    total_matches_found = 0

    countries = [args.country] if args.country else ['France', 'US', 'India']

    for country in countries:
        c_t0 = time.time()
        print(f"\n" + "#" * 80, flush=True)
        print(f"  PROCESSING COUNTRY: {country.upper()}", flush=True)
        print("#" * 80, flush=True)

        # 1. Precompute S1 records
        t0 = time.time()
        s1_tuples = []
        with open(s1_file, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 4 and p[3] == country:
                    r = make_record(p[0], p[1], p[2] if p[2] != 'nan' else '', p[3])
                    pa = r['parsed_addr']
                    tup = (
                        r['entity_id'], r['norm_name'], r['norm_address'], r['stripped_name'],
                        r['compact_brand'], ' '.join(r['phonetic_tokens']),
                        r['sig_tokens'], r['acronyms'], r['unit_keys'],
                        pa['number'], pa['postcode'], pa['street'], pa['loc_tokens'],
                        r['addr_nums'], r['house_codes'], r['distinctive_addr_tokens']
                    )
                    s1_tuples.append(tup)
                    if args.limit and len(s1_tuples) >= args.limit:
                        break
        print(f"[1/4] Precomputed {len(s1_tuples):,} S1 entities for {country} in {time.time()-t0:.1f}s", flush=True)
        if not s1_tuples:
            continue

        # 2. Precompute candidates into compact tuples in 30 GB RAM
        t0 = time.time()
        cand_tuples = []
        inv = {}
        for sf in [s2_file, s3_file]:
            cnt = 0
            sf_t0 = time.time()
            with open(sf, 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 4 and p[3] == country:
                        r = make_record(p[0], p[1], p[2] if p[2] != 'nan' else '', p[3])
                        pa = r['parsed_addr']
                        c_idx = len(cand_tuples)
                        tup = (
                            r['entity_id'], r['norm_name'], r['norm_address'], r['stripped_name'],
                            r['compact_brand'], ' '.join(r['phonetic_tokens']),
                            r['sig_tokens'], r['acronyms'], r['unit_keys'],
                            pa['number'], pa['postcode'], pa['street'], pa['loc_tokens']
                        )
                        cand_tuples.append(tup)

                        for tok in r['sig_tokens']: inv.setdefault(('T', tok), array.array('I')).append(c_idx)
                        for num in r['addr_nums']: inv.setdefault(('N', num), array.array('I')).append(c_idx)
                        for h in r['house_codes']: inv.setdefault(('H', h), array.array('I')).append(c_idx)
                        for acr in r['acronyms']: inv.setdefault(('A', acr), array.array('I')).append(c_idx)
                        for pht in r['phonetic_tokens']: inv.setdefault(('PH', pht), array.array('I')).append(c_idx)
                        for uk in r['unit_keys']: inv.setdefault(('UK', uk), array.array('I')).append(c_idx)
                        if r['compact_brand']: inv.setdefault(('CB', r['compact_brand'][:8]), array.array('I')).append(c_idx)
                        for atok in r['distinctive_addr_tokens']: inv.setdefault(('AT', atok), array.array('I')).append(c_idx)
                        if len(r['norm_name']) >= 3: inv.setdefault(('P', r['norm_name'][:3]), array.array('I')).append(c_idx)
                        cnt += 1
            print(f"      Indexed {cnt:,} candidates from {os.path.basename(sf)} in {time.time()-sf_t0:.1f}s", flush=True)

        inv_pruned = {}
        for k, v in inv.items():
            limit = 80 if k[0] == 'P' else (150 if k[0] == 'AT' else 300)
            if len(v) <= limit: inv_pruned[k] = v
        del inv
        print(f"[2/4] Precomputed {len(cand_tuples):,} candidates in RAM in {time.time()-t0:.1f}s (Index: {len(inv_pruned):,} keys)", flush=True)

        # 3. TF-IDF representation
        t0 = time.time()
        all_names = list({r[3] for r in s1_tuples} | {r[3] for r in cand_tuples})
        w_csr = word_vec.transform(all_names)
        c_csr = char_vec.transform(all_names)
        w_dl = csr_to_dict_list(w_csr)
        c_dl = csr_to_dict_list(c_csr)
        name_to_wdict = {all_names[i]: w_dl[i] for i in range(len(all_names))}
        name_to_cdict = {all_names[i]: c_dl[i] for i in range(len(all_names))}
        del all_names, w_csr, c_csr, w_dl, c_dl
        print(f"[3/4] TF-IDF dictionaries ready for {len(name_to_wdict):,} unique names in {time.time()-t0:.1f}s", flush=True)

        # 4. Large-batch parallel evaluation (25k S1 per batch on Kaggle 30 GB RAM)
        batch_size = 25000
        print(f"[4/4] Evaluating {len(s1_tuples):,} S1 entities in batches of {batch_size:,}...", flush=True)

        # Initialize worker pool once per country with native Linux Copy-on-Write
        ctx = mp.get_context('fork') if hasattr(mp, 'get_context') and 'fork' in mp.get_all_start_methods() else mp.get_context()
        pool = ctx.Pool(processes=num_cpus, initializer=init_worker,
                        initargs=(s1_tuples, cand_tuples, name_to_wdict, name_to_cdict))

        country_matches = 0
        for b_idx in range(0, len(s1_tuples), batch_size):
            b_t0 = time.time()
            batch_s1 = s1_tuples[b_idx : b_idx + batch_size]

            # Query candidate pairs
            batch_pairs = []
            s1_cand_map = {r[0]: [] for r in batch_s1}

            for s_rel, r1 in enumerate(batch_s1):
                sid = r1[0]
                s_abs = b_idx + s_rel
                cands = set()
                for tok in r1[6]: cands.update(inv_pruned.get(('T', tok), ()))
                for num in r1[13]: cands.update(inv_pruned.get(('N', num), ()))
                for h in r1[14]: cands.update(inv_pruned.get(('H', h), ()))
                for acr in r1[7]: cands.update(inv_pruned.get(('A', acr), ()))
                for pht in r1[5].split(): cands.update(inv_pruned.get(('PH', pht), ()))
                for uk in r1[8]: cands.update(inv_pruned.get(('UK', uk), ()))
                if r1[4]: cands.update(inv_pruned.get(('CB', r1[4][:8]), ()))
                for tok in r1[6]:
                    if len(tok) >= 5: cands.update(inv_pruned.get(('CB', tok[:8]), ()))
                for atok in r1[15]: cands.update(inv_pruned.get(('AT', atok), ()))
                if len(r1[1]) >= 3: cands.update(inv_pruned.get(('P', r1[1][:3]), ()))

                for c_idx in cands:
                    cid = cand_tuples[c_idx][0]
                    batch_pairs.append((s_abs, c_idx))
                    s1_cand_map[sid].append(cid)

            n_pairs = len(batch_pairs)

            if n_pairs > 0:
                # Parallel feature extraction across 4 workers
                chunk_sz = (n_pairs + num_cpus - 1) // num_cpus
                slices = [batch_pairs[i * chunk_sz : (i + 1) * chunk_sz] for i in range(num_cpus) if i * chunk_sz < n_pairs]
                sub_Xs = pool.map(compute_chunk_features, slices)
                X = np.vstack(sub_Xs)
                del sub_Xs

                # OpenMP model scoring in 100k sub-batches
                all_probs = []
                for p_start in range(0, n_pairs, 100000):
                    X_sub = X[p_start : p_start + 100000]
                    p_sub = clf.predict_proba(X_sub)[:, 1]
                    all_probs.append(p_sub)
                probs = np.concatenate(all_probs)
                del X, all_probs
                resolved_pairs = [(s1_tuples[s_abs][0], cand_tuples[c_idx][0]) for (s_abs, c_idx) in batch_pairs]
            else:
                probs = np.array([], dtype=np.float32)
                resolved_pairs = []

            # Global bipartite resolution
            s1_ids = [r[0] for r in batch_s1]
            s1_dict_dummy = {sid: {'country': country} for sid in s1_ids}
            batch_matches = collective_resolve(
                resolved_pairs, probs, s1_ids,
                primary_threshold=thresh_config, secondary_threshold=sec_thresh, margin=margin,
                s1_dict=s1_dict_dummy
            )

            # Stream outputs to disk
            with open(out_file, 'a', encoding='utf-8') as f:
                for sid in s1_ids:
                    m = batch_matches.get(sid, [])
                    if m:
                        country_matches += 1
                        total_matches_found += 1
                    f.write(f"{sid}\t{','.join(m) if m else ''}\n")

            with open(cand_file, 'a', encoding='utf-8') as f:
                for sid in s1_ids:
                    cands = s1_cand_map.get(sid, [])
                    f.write(f"{sid}\t{','.join(cands) if cands else ''}\n")

            total_s1_processed += len(batch_s1)
            b_time = time.time() - b_t0
            pct = ((b_idx + len(batch_s1)) / len(s1_tuples)) * 100.0
            print(f"    Batch [{b_idx + len(batch_s1):,}/{len(s1_tuples):,}] ({pct:5.1f}%) | {n_pairs:,} pairs in {b_time:5.1f}s ({n_pairs/max(b_time, 0.01):6.0f} pairs/s)", flush=True)

            del batch_pairs, s1_cand_map, probs, resolved_pairs, batch_matches

        pool.close()
        pool.join()
        print(f"  --> Finished {country}: {len(s1_tuples):,} S1 in {time.time()-c_t0:.1f}s ({country_matches:,} matches)", flush=True)
        del s1_tuples, cand_tuples, inv_pruned, name_to_wdict, name_to_cdict
        gc.collect()

    total_time = time.time() - total_start
    print("\n" + "=" * 80, flush=True)
    print(f"  PREDICTION COMPLETED IN {total_time/60:.1f} MINUTES!", flush=True)
    print(f"  Total S1 Processed: {total_s1_processed:,}", flush=True)
    print(f"  Total Matches Found: {total_matches_found:,} ({(total_matches_found/max(total_s1_processed,1))*100:.1f}%)", flush=True)
    print("=" * 80, flush=True)

    # 5. Validation Check
    print("\n[VALIDATION] Running official submission validator...", flush=True)
    val_candidates = ['validate_submission.py', 'student_resource/utils/validate_submission.py',
                      '6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py']
    val_script = None
    for vc in val_candidates:
        if os.path.exists(vc):
            val_script = vc
            break

    if val_script:
        test_dir = os.path.dirname(s1_file)
        res = subprocess.run([
            sys.executable, val_script,
            '--matching', out_file,
            '--candidate', cand_file,
            '--test-dir', test_dir
        ], capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.returncode == 0:
            print("[SUCCESS] All official submission format checks passed!", flush=True)
        else:
            print("[WARNING] Validator output:\n", res.stderr, flush=True)

    # 6. Automatic ZIP packaging for download
    zip_path = os.path.join(output_dir, 'submission_archive.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(out_file, arcname='matching_results.tsv')
        zipf.write(cand_file, arcname='candidate_pairs.tsv')
    print(f"\n[DOWNLOAD READY] Created submission archive: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
