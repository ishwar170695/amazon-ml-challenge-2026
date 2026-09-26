"""
Full-Featured Real-Data Pipeline v3.1 (Hard-Negative Trained)
============================================================
Key Fix:
- Step 3 now mines HARD NEGATIVES directly from the inverted index (sharing tokens/numbers/acronyms).
  This eliminates the 8,000+ false merges caused by training solely on random country negatives.
- Added full address Levenshtein (alv) and missing address flag (mad).
- RapidFuzz C++ + dict TF-IDF keeps execution under 2 minutes.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, re, time, random, pickle, unicodedata, json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
import anyascii

random.seed(42); np.random.seed(42)

# ================================================================
# TEXT NORMALIZATION & PREPROCESSING
# ================================================================
LEGAL_SUFFIXES_REGEX = r'\b(corp|corporation|incorporated|inc|ltd|limited|pvt|private|llc|llp|gmbh|ag|sa|sarl|sas|sasu|plc|bv|nv|spa|srl|sl|cie|co|company)\b'
STOPWORDS = {'and', 'the', '&', 'of', 'in', 'at', 'on', 'for', 'by', 'corp', 'limited', 'pvt', 'ltd', 'inc', 'llc'}

ADDRESS_ABBREVS = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bln\b': 'lane', r'\bbldg\b': 'building',
    r'\bapt\b': 'apartment', r'\bste\b': 'suite', r'\bfl\b': 'floor',
    r'\bdr\b': 'drive', r'\bct\b': 'court', r'\bpkwy\b': 'parkway',
    r'\bopp\b': 'opposite', r'\bnear\b': 'near',
}

LEGAL_MAP = {
    r'\bpvt\s*ltd\b': 'corp', r'\bprivate\s*limited\b': 'corp',
    r'\blimited\b': 'corp', r'\bltd\b': 'corp', r'\bllc\b': 'corp',
    r'\binc\b': 'corp', r'\bincorporated\b': 'corp',
    r'\bcorporation\b': 'corp', r'\bcorp\b': 'corp',
    r'\bco\b': 'corp', r'\bcompany\b': 'corp',
    r'\bsarl\b': 'corp', r'\bgmbh\b': 'corp', r'\bllp\b': 'corp',
    r'\bsa\b': 'corp', r'\bsas\b': 'corp',
}

def strip_accents(t):
    if not isinstance(t, str): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')

def normalize_text(t, is_address=False):
    if not isinstance(t, str): return ""
    t = strip_accents(t.lower().strip())
    t = re.sub(r'\b([a-z])\.(?:\s*([a-z])\.?)+', lambda m: m.group(0).replace('.','').replace(' ',''), t)
    t = re.sub(r'[^\w\s]', ' ', t)
    for pat, rep in (ADDRESS_ABBREVS if is_address else LEGAL_MAP).items():
        t = re.sub(pat, rep, t)
    return re.sub(r'\s+', ' ', t).strip()

def sig_tokens(nn):
    return {t for t in nn.split() if len(t) >= 3 and t not in STOPWORDS}

def strip_legal(n):
    return re.sub(r'\s+', ' ', re.sub(LEGAL_SUFFIXES_REGEX, '', n.lower())).strip()

def gen_acronyms(nn):
    cl = strip_legal(nn)
    at = [t for t in cl.split() if t not in STOPWORDS and t.isalpha()]
    acrs = set()
    if len(at) >= 2:
        a = ''.join(t[0] for t in at)
        if 2 <= len(a) <= 6: acrs.add(a)
    elif len(at) == 1 and 2 <= len(at[0]) <= 5:
        acrs.add(at[0])
    return acrs

def parse_addr(addr):
    if not isinstance(addr, str) or not addr.strip():
        return {'number': '', 'postcode': '', 'street': '', 'locality': '', 'loc_tokens': set()}
    a = addr.strip()
    pm = re.search(r'\b\d{5,6}\b', a)
    pc = pm.group(0) if pm else ''
    ca = re.sub(r'\b\d{5,6}\b', '', a).strip()
    num = ''
    pm2 = re.search(r'\b(?:no\.?|plot|door|bldg|flat|h\.?no\.?)\s*[:#-]?\s*([0-9a-zA-Z\-/]+)\b', ca, re.I)
    if pm2:
        num = pm2.group(1).strip().lower()
        ca = ca[:pm2.start()] + ca[pm2.end():]
    else:
        lm = re.match(r'^(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', ca, re.I)
        if lm:
            num = lm.group(1).strip().lower()
            ca = ca[lm.end():].strip(', ')
        else:
            pts = [p.strip() for p in ca.split(',') if p.strip()]
            if pts:
                tm = re.search(r'\b(\d+\s*(?:bis|ter|[a-zA-Z])?)\s*$', pts[0], re.I)
                if tm:
                    num = tm.group(1).strip().lower()
                    pts[0] = pts[0][:tm.start()].strip()
                    ca = ', '.join(pts)
    pts = [p.strip() for p in ca.split(',') if p.strip()]
    loc = pts[-1].lower() if len(pts) > 1 else ''
    st = ', '.join(pts[:-1]).lower() if len(pts) > 1 else (pts[0].lower() if pts else '')
    loc_tokens = {w for w in re.sub(r'[^\w\s]', ' ', loc).split() if len(w) >= 3 and w not in STOPWORDS}
    return {'number': num, 'postcode': pc, 'street': st, 'locality': loc, 'loc_tokens': loc_tokens}

def get_addr_numbers(t):
    if not isinstance(t, str): return set()
    return set(re.findall(r'(?<!\d)\d{2,6}(?!\d)', t))

def get_house_codes(t):
    if not isinstance(t, str): return set()
    return set(re.findall(r'\b[a-zA-Z]-?\d{1,4}\b', t.lower()))

def make_char3_set(s):
    if not s or len(s) < 3: return set()
    return {s[i:i+3] for i in range(len(s)-2)}

def make_record(eid, name, addr, country):
    name_ascii = anyascii.anyascii(name) if any(ord(c) > 127 for c in name) else name
    addr_ascii = anyascii.anyascii(addr) if any(ord(c) > 127 for c in addr) else addr
    
    nn = normalize_text(name_ascii)
    na = normalize_text(addr_ascii, is_address=True)
    pa = parse_addr(addr_ascii)
    
    return {
        'entity_id': eid,
        'country': country,
        'norm_name': nn,
        'norm_address': na,
        'sig_tokens': sig_tokens(nn),
        'acronyms': gen_acronyms(nn),
        'parsed_addr': pa,
        'stripped_name': strip_legal(nn),
        'addr_nums': get_addr_numbers(addr_ascii),
        'house_codes': get_house_codes(addr_ascii),
        'c3_name': make_char3_set(nn),
        'c3_addr': make_char3_set(na),
    }

# ================================================================
# FEATURE EXTRACTION (RapidFuzz C++ + Dict TF-IDF)
# ================================================================
FCOLS = ['ntj', 'nts', 'nlv', 'nc3', 'twc', 'tcc', 'lex', 'acr', 'nsm', 'mnm', 'pcm', 'mpc', 'ljc', 'slv', 'ac3', 'alv', 'mad', 'nld']

def set_jaccard(a, b):
    if not a or not b: return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def csr_to_dict_list(mat):
    indptr = mat.indptr
    indices = mat.indices
    data = mat.data
    return [dict(zip(indices[indptr[i]:indptr[i+1]], data[indptr[i]:indptr[i+1]])) for i in range(mat.shape[0])]

def feats_batch(pairs, s1d, cd, s1_wdict, c_wdict, s1_cdict, c_cdict):
    out = []
    for s1id, cid in pairs:
        r1, r2 = s1d[s1id], cd[cid]
        n1, n2 = r1['norm_name'], r2['norm_name']
        a1, a2 = r1['norm_address'], r2['norm_address']
        p1, p2 = r1['parsed_addr'], r2['parsed_addr']
        
        # 1. Name Token Jaccard
        ntj = set_jaccard(r1['sig_tokens'], r2['sig_tokens'])
        
        # 2 & 3. RapidFuzz token_sort_ratio and ratio (C++ accelerated)
        nts = fuzz.token_sort_ratio(n1, n2) / 100.0
        nlv = fuzz.ratio(n1, n2) / 100.0
        
        # 4. Name Char 3-gram Jaccard (precomputed sets)
        nc3 = set_jaccard(r1['c3_name'], r2['c3_name'])
        
        # 5 & 6. TF-IDF Cosines (dict dot products: 0.6 microseconds each!)
        d1 = s1_wdict.get(s1id, {})
        d2 = c_wdict.get(cid, {})
        twc = sum(v * d2[k] for k, v in d1.items() if k in d2) if d1 and d2 else 0.0
        
        cd1 = s1_cdict.get(s1id, {})
        cd2 = c_cdict.get(cid, {})
        tcc = sum(v * cd2[k] for k, v in cd1.items() if k in cd2) if cd1 and cd2 else 0.0
        
        # 7. Legal-stripped Exact Match
        lex = 1.0 if r1['stripped_name'] and r1['stripped_name'] == r2['stripped_name'] else 0.0
        
        # 8. Acronym match
        acr = 1.0 if r1['acronyms'] & r2['acronyms'] else 0.0
        
        # 9 & 10. Number match
        if p1['number'] and p2['number']:
            nsm = 1.0 if p1['number'] == p2['number'] else -1.0
            mnm = 0.0
        else:
            nsm = 0.0; mnm = 1.0
            
        # 11 & 12. Postcode / PIN match
        if p1['postcode'] and p2['postcode']:
            if p1['postcode'] == p2['postcode']:
                pcm = 1.0
            elif p1['postcode'][:3] == p2['postcode'][:3]:
                pcm = 0.5
            else:
                pcm = -1.0
            mpc = 0.0
        else:
            pcm = 0.0; mpc = 1.0
            
        # 13. Locality Jaccard
        ljc = set_jaccard(p1['loc_tokens'], p2['loc_tokens'])
        
        # 14. Street Levenshtein (RapidFuzz C++)
        st1, st2 = p1['street'], p2['street']
        slv = (fuzz.ratio(st1, st2) / 100.0) if st1 and st2 else 0.0
        
        # 15. Address Char 3-gram Jaccard (precomputed sets)
        ac3 = set_jaccard(r1['c3_addr'], r2['c3_addr'])
        
        # 16. Full address Levenshtein ratio
        alv = (fuzz.ratio(a1, a2) / 100.0) if a1 and a2 else 0.0
        
        # 17. Missing address flag
        mad = 1.0 if not a1 or not a2 else 0.0
        
        # 18. Name length difference ratio
        max_l = max(len(n1), len(n2), 1)
        nld = abs(len(n1) - len(n2)) / max_l
        
        out.append([ntj, nts, nlv, nc3, twc, tcc, lex, acr, nsm, mnm, pcm, mpc, ljc, slv, ac3, alv, mad, nld])
    
    return np.array(out, dtype=np.float32)

# ================================================================
# EVALUATION & COLLECTIVE RESOLUTION
# ================================================================
def macro_f05(gt, pred):
    scores = []
    for s1, tr in gt.items():
        ts = set(tr)
        ps = set(pred.get(s1, []))
        if not ts:
            scores.append(1.0 if not ps else 0.0)
        elif not ps:
            scores.append(0.0)
        else:
            tp = len(ts & ps)
            p = tp / len(ps)
            r = tp / len(ts)
            scores.append((1.25 * p * r) / (0.25 * p + r) if p + r > 0 else 0.0)
    return float(np.mean(scores))

def collective_resolve(pairs, probs, eids, pt=0.5, st=0.6, mm=0.20):
    s1c = {s: [] for s in eids}
    cc = {}
    for (s1, c), p in zip(pairs, probs):
        if s1 in s1c and p >= pt:
            s1c[s1].append((c, p))
            cc.setdefault(c, []).append((p, s1))
    
    cw = {}
    for c, cl in cc.items():
        cl.sort(reverse=True, key=lambda x: x[0])
        cw[c] = cl[0][1]
        
    fm = {s: [] for s in eids}
    for s1, cds in s1c.items():
        if not cds: continue
        cds.sort(reverse=True, key=lambda x: x[1])
        tc, tp = cds[0]
        if cw.get(tc) == s1:
            fm[s1].append(tc)
        for c, p in cds[1:]:
            if cw.get(c) == s1 and p >= st and (tp - p) <= mm:
                fm[s1].append(c)
    return fm

# ================================================================
# MAIN EXECUTION PIPELINE
# ================================================================
def main():
    T0 = time.time()
    print("=" * 75, flush=True)
    print("  PIPELINE v3.1: Hard-Negative Mining + RapidFuzz C++ + Dict TF-IDF", flush=True)
    print("=" * 75, flush=True)

    N_TR, N_VL = 20000, 5000
    TOT = N_TR + N_VL

    # STEP 1: Load S1 + Ground Truth
    t = time.time()
    print(f"\n[1/6] Loading {TOT:,} S1 entities and ground truth...", flush=True)
    s1d = {}
    with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if len(p) >= 4:
                s1d[p[0]] = make_record(p[0], p[1], p[2] if p[2] != 'nan' else '', p[3])
            if len(s1d) >= TOT: break
            
    s1ids = list(s1d.keys())
    tr_ids = set(s1ids[:N_TR])
    vl_ids = set(s1ids[N_TR:])
    
    gt = {}
    needed = set()
    with open('dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.strip().split('\t')
            if p and p[0] in s1d:
                m = [x.strip() for x in p[1].split(',') if x.strip() and x.strip() != 'nan'] if len(p) > 1 else []
                gt[p[0]] = m
                needed.update(m)
                
    print(f"  Loaded {len(s1d):,} S1 entities | {len(needed):,} targets in GT ({time.time()-t:.1f}s)", flush=True)

    # STEP 2: Load S2 / S3 Candidates
    t = time.time()
    print(f"\n[2/6] Loading candidates from S2 and S3...", flush=True)
    cd = {}
    for sf, rem in [('dataset/train/train_source2.tsv', {x for x in needed if x.startswith('S2-')}),
                     ('dataset/train/train_source3.tsv', {x for x in needed if x.startswith('S3-')})]:
        bg = 0
        with open(sf, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 4:
                    cid = p[0]
                    isn = cid in rem
                    if isn or bg < 20000:
                        cd[cid] = make_record(cid, p[1], p[2] if p[2] != 'nan' else '', p[3])
                        if isn: rem.discard(cid)
                        else: bg += 1
                    if not rem and bg >= 20000: break
                    
    print(f"  Total Candidate Pool: {len(cd):,} records ({time.time()-t:.1f}s)", flush=True)

    # STEP 3: Build Inverted Index First (Shared for Training Hard Negatives + Validation)
    t = time.time()
    print(f"\n[3/6] Building high-recall inverted index across candidate pool...", flush=True)
    inv = {}
    for cid, r in cd.items():
        c = r['country']
        for tok in r['sig_tokens']:
            inv.setdefault((c, 'T', tok), []).append(cid)
        for num in r['addr_nums']:
            inv.setdefault((c, 'N', num), []).append(cid)
        for h in r['house_codes']:
            inv.setdefault((c, 'H', h), []).append(cid)
        for acr in r['acronyms']:
            inv.setdefault((c, 'A', acr), []).append(cid)
        pref = r['norm_name'][:3]
        if len(pref) >= 3:
            inv.setdefault((c, 'P', pref), []).append(cid)

    # Pruning: tokens/numbers/houses pruned at 300; prefixes pruned at 80
    inv_pruned = {}
    for k, v in inv.items():
        limit = 80 if k[1] == 'P' else 300
        if len(v) <= limit:
            inv_pruned[k] = v
    print(f"  Inverted index built with {len(inv_pruned):,} active posting lists ({time.time()-t:.1f}s)", flush=True)

    # Function to query candidates for an entity
    def query_blocking(r):
        c = r['country']
        cands = set()
        for tok in r['sig_tokens']:
            cands.update(inv_pruned.get((c, 'T', tok), []))
        for num in r['addr_nums']:
            cands.update(inv_pruned.get((c, 'N', num), []))
        for h in r['house_codes']:
            cands.update(inv_pruned.get((c, 'H', h), []))
        for acr in r['acronyms']:
            cands.update(inv_pruned.get((c, 'A', acr), []))
        pref = r['norm_name'][:3]
        if len(pref) >= 3:
            cands.update(inv_pruned.get((c, 'P', pref), []))
        return cands

    # STEP 4: Build Training Pairs with HARD NEGATIVE MINING from Inverted Index
    t = time.time()
    print(f"\n[4/6] Mining hard negatives from blocking index for training...", flush=True)
    cands_by_country = {}
    for cid, r in cd.items():
        cands_by_country.setdefault(r['country'], []).append(cid)

    train_pairs = []
    hard_neg_count = 0
    rand_neg_count = 0

    for sid in tr_ids:
        r = s1d[sid]
        country = r['country']
        positives = [m for m in gt.get(sid, []) if m in cd]
        for cid in positives:
            train_pairs.append((sid, cid))
        
        pos_set = set(positives)
        blocking_cands = query_blocking(r)
        hard_negs = [c for c in blocking_cands if c not in pos_set]
        
        # Sample up to 10 hard negatives from blocking
        if len(hard_negs) > 10:
            sampled_hard = random.sample(hard_negs, 10)
        else:
            sampled_hard = hard_negs
            
        for cid in sampled_hard:
            train_pairs.append((sid, cid))
            hard_neg_count += 1
            pos_set.add(cid)
            
        # Also sample up to 2 random country negatives
        pool = cands_by_country.get(country, [])
        for _ in range(2):
            if pool:
                rc = random.choice(pool)
                if rc not in pos_set:
                    train_pairs.append((sid, rc))
                    rand_neg_count += 1
                    pos_set.add(rc)

    print(f"  Training pairs: {len(train_pairs):,} (Positives: {len([p for p in train_pairs if p[1] in gt.get(p[0], [])]):,}, Hard Negs: {hard_neg_count:,}, Rand Negs: {rand_neg_count:,}) ({time.time()-t:.1f}s)", flush=True)

    # Query validation candidates
    t = time.time()
    val_pairs = []
    val_blocking_stats = {}
    for sid in vl_ids:
        r = s1d[sid]
        c = r['country']
        cands = query_blocking(r)

        true_set = set(gt.get(sid, [])) & set(cd.keys())
        cap = len(true_set & cands)
        tot = len(true_set)
        if c not in val_blocking_stats: val_blocking_stats[c] = [0, 0]
        val_blocking_stats[c][0] += cap
        val_blocking_stats[c][1] += tot

        for cid in cands:
            val_pairs.append((sid, cid))

    print(f"\n  --- Per-Country Blocking Recall (Validation Set) ---", flush=True)
    tc, tt = 0, 0
    for c in sorted(val_blocking_stats):
        cap, tot = val_blocking_stats[c]
        tc += cap; tt += tot
        pct = (cap / tot * 100) if tot else 0.0
        print(f"    {c:10s}: {cap:5d} / {tot:5d} ({pct:6.2f}%)", flush=True)
    print(f"    {'OVERALL':10s}: {tc:5d} / {tt:5d} ({tc/tt*100:6.2f}%)", flush=True)
    print(f"  Validation pairs: {len(val_pairs):,} ({time.time()-t:.1f}s)", flush=True)

    # STEP 5: TF-IDF & RapidFuzz Feature Extraction (Accelerated)
    t = time.time()
    print(f"\n[5/6] Fitting batch TF-IDF & converting to fast dicts...", flush=True)
    s1_uniq = list({s for s, _ in train_pairs + val_pairs})
    c_uniq = list({c for _, c in train_pairs + val_pairs})
    s1n = [s1d[s]['norm_name'] for s in s1_uniq]
    cn = [cd[c]['norm_name'] for c in c_uniq]
    alln = s1n + cn

    wt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95).fit(alln)
    ct = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_df=0.95).fit(alln)
    del alln

    s1wv = wt.transform(s1n); s1cv = ct.transform(s1n)
    cwv = wt.transform(cn); ccv = ct.transform(cn)
    del s1n, cn

    s1_wdict = dict(zip(s1_uniq, csr_to_dict_list(s1wv)))
    s1_cdict = dict(zip(s1_uniq, csr_to_dict_list(s1cv)))
    c_wdict = dict(zip(c_uniq, csr_to_dict_list(cwv)))
    c_cdict = dict(zip(c_uniq, csr_to_dict_list(ccv)))
    del s1wv, s1cv, cwv, ccv
    print(f"  Fast sparse dicts created in {time.time()-t:.1f}s", flush=True)

    t_feat = time.time()
    print(f"  Extracting {len(train_pairs):,} train features...", flush=True)
    Xtr = feats_batch(train_pairs, s1d, cd, s1_wdict, c_wdict, s1_cdict, c_cdict)
    ytr = np.array([1 if c in gt.get(s, []) else 0 for s, c in train_pairs], dtype=np.int32)
    print(f"  Xtr extracted: {Xtr.shape} in {time.time()-t_feat:.1f}s (Positives: {np.sum(ytr):,}, Negatives: {len(ytr)-np.sum(ytr):,})", flush=True)

    # Train LightGBM with real hard negatives
    t_train = time.time()
    print(f"  Training LightGBM model on hard negatives...", flush=True)
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=8,
                         min_child_samples=20, random_state=42, class_weight='balanced',
                         n_jobs=-1, verbose=-1, subsample=0.8, colsample_bytree=0.8)
    clf.fit(Xtr, ytr)
    print(f"  LightGBM trained in {time.time()-t_train:.1f}s", flush=True)

    print(f"\n  --- Feature Importances ---", flush=True)
    imp = pd.Series(clf.feature_importances_, index=FCOLS).sort_values(ascending=False)
    for c, v in imp.items():
        print(f"    {c:6s}: {v:5d} ({v/imp.sum()*100:5.1f}%)", flush=True)

    # STEP 6: Validation, Fine Threshold Sweep & Collective Resolution
    t_val = time.time()
    print(f"\n[6/6] Extracting {len(val_pairs):,} val features & evaluating...", flush=True)
    Xvl = feats_batch(val_pairs, s1d, cd, s1_wdict, c_wdict, s1_cdict, c_cdict)
    print(f"  Val features extracted in {time.time()-t_val:.1f}s", flush=True)
    vp = clf.predict_proba(Xvl)[:, 1]

    vgt = {s: gt.get(s, []) for s in vl_ids}

    print(f"\n  --- Fine Threshold & Collective Resolution Sweep ---", flush=True)
    best_pw_t, best_pw_f05 = 0.5, 0.0
    best_coll_t, best_coll_f05 = 0.5, 0.0
    sweep_results = []

    for th in np.arange(0.20, 0.95, 0.05):
        th = round(th, 2)
        # Pairwise
        pw = {s: [] for s in vl_ids}
        for (s, c), p in zip(val_pairs, vp):
            if p >= th: pw[s].append(c)
        pf = macro_f05(vgt, pw)
        
        # Collective bipartite
        cf_preds = collective_resolve(val_pairs, vp, list(vl_ids), pt=th, st=th + 0.10, mm=0.20)
        cf = macro_f05(vgt, cf_preds)
        
        marker = ""
        if pf > best_pw_f05:
            best_pw_f05 = pf; best_pw_t = th
        if cf > best_coll_f05:
            best_coll_f05 = cf; best_coll_t = th; marker = " <=== BEST COLL"
            
        print(f"    Thresh={th:.2f} | Pairwise F0.5={pf:.4f} | Collective F0.5={cf:.4f}{marker}", flush=True)
        sweep_results.append({'threshold': th, 'pairwise_f05': pf, 'collective_f05': cf})

    print(f"\n  =======================================================", flush=True)
    print(f"  >>> BEST PAIRWISE : Thresh={best_pw_t:.2f} | F0.5={best_pw_f05:.4f}", flush=True)
    print(f"  >>> BEST COLLECTIVE: Thresh={best_coll_t:.2f} | F0.5={best_coll_f05:.4f}", flush=True)
    print(f"  =======================================================", flush=True)

    # Detailed Error Taxonomy
    best_preds = collective_resolve(val_pairs, vp, list(vl_ids), pt=best_coll_t, st=best_coll_t + 0.10, mm=0.20)
    fps, fns = [], []
    country_scores = {}
    for sid in vl_ids:
        c = s1d[sid]['country']
        if c not in country_scores: country_scores[c] = {'gt': {}, 'pred': {}}
        country_scores[c]['gt'][sid] = vgt[sid]
        country_scores[c]['pred'][sid] = best_preds.get(sid, [])
        
        ts = set(vgt[sid])
        ps = set(best_preds.get(sid, []))
        for cand in ps - ts: fps.append((sid, cand))
        for cand in ts - ps:
            if cand in cd: fns.append((sid, cand))

    print(f"\n  --- Per-Country Evaluation at Best Threshold ({best_coll_t:.2f}) ---", flush=True)
    for c in sorted(country_scores):
        cF = macro_f05(country_scores[c]['gt'], country_scores[c]['pred'])
        print(f"    {c:10s}: F0.5 = {cF:.4f}", flush=True)

    print(f"\n  False Positives: {len(fps):,} | False Negatives: {len(fns):,}", flush=True)
    if fps:
        print(f"\n  Sample False Merges (FP):", flush=True)
        for s, c in fps[:4]:
            print(f"    S1: {s1d[s]['norm_name']} @ {s1d[s]['norm_address']}", flush=True)
            print(f"    FP: {cd[c]['norm_name']} @ {cd[c]['norm_address']}", flush=True)
            print("    " + "-"*40, flush=True)
    if fns:
        print(f"\n  Sample Misses (FN):", flush=True)
        for s, c in fns[:4]:
            print(f"    S1: {s1d[s]['norm_name']} @ {s1d[s]['norm_address']}", flush=True)
            print(f"    FN: {cd[c]['norm_name']} @ {cd[c]['norm_address']}", flush=True)
            print("    " + "-"*40, flush=True)

    # Save artifacts
    os.makedirs('artifacts', exist_ok=True)
    result_data = {
        'best_pairwise_threshold': best_pw_t,
        'best_pairwise_f05': best_pw_f05,
        'best_collective_threshold': best_coll_t,
        'best_collective_f05': best_coll_f05,
        'n_fps': len(fps),
        'n_fns': len(fns),
        'blocking_recall': {c: [s[0], s[1], round(s[0]/s[1]*100, 2) if s[1] else 0] for c, s in val_blocking_stats.items()},
        'per_country_f05': {c: round(macro_f05(country_scores[c]['gt'], country_scores[c]['pred']), 4) for c in country_scores},
        'sweep': sweep_results,
        'total_runtime_s': round(time.time() - T0, 1)
    }
    with open('artifacts/result_v3.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    with open('artifacts/model_v3.pkl', 'wb') as f:
        pickle.dump({'model': clf, 'word_tfidf': wt, 'char_tfidf': ct, 'best_threshold': best_coll_t}, f)

    print(f"\n{'='*75}\n  PIPELINE FINISHED IN {time.time()-T0:.1f}s. Results saved to artifacts/result_v3.json\n{'='*75}", flush=True)

if __name__ == '__main__':
    main()
