"""
3-Way Split Evaluation + Leetspeak Normalization + French Open-Set Audit
========================================================================
1. Strict 3-Way Split:
   - Train (20,000 S1 entities): model trained with inverted-index hard negatives
   - Val   (2,500 S1 entities) : threshold tuned here (T_opt)
   - Test  (2,500 S1 entities) : evaluated ONLY at T_opt (strictly held-out, zero tuning leakage)
2. Normalization Fixes:
   - Leetspeak digit repair: 'de1hi' -> 'delhi', 'pear1' -> 'pearl'
   - Address parser bugfix: 'North'/'Nouvelle'/'Nord' no longer misparsed as house numbers
   - Address comma-preservation: locality and street parsed before comma stripping
   - International & French legal forms: eurl, sci, snc, gie, sasu, sarl, sas
3. French Open-Set Stress-Test:
   - Synthetic French adversarial benchmark evaluated with the trained model
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
# NORMALIZATION & PARSING
# ================================================================
LEGAL_SUFFIXES_REGEX = r'\b(corp|corporation|incorporated|inc|ltd|limited|pvt|private|llc|llp|gmbh|ag|sa|sarl|sas|sasu|plc|bv|nv|spa|srl|sl|cie|co|company|eurl|sci|snc|gie)\b'
STOPWORDS = {'and', 'the', '&', 'of', 'in', 'at', 'on', 'for', 'by', 'corp', 'limited', 'pvt', 'ltd', 'inc', 'llc', 'des', 'les', 'aux', 'sur', 'sous', 'sci', 'eurl', 'sarl', 'sasu', 'sas'}

ADDRESS_ABBREVS = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bln\b': 'lane', r'\bbldg\b': 'building',
    r'\bapt\b': 'apartment', r'\bste\b': 'suite', r'\bfl\b': 'floor',
    r'\bdr\b': 'drive', r'\bct\b': 'court', r'\bpkwy\b': 'parkway',
    r'\bopp\b': 'opposite', r'\bnear\b': 'near',
    r'\br\b': 'street', r'\brue\b': 'street', r'\bav\b': 'avenue',
    r'\bbd\b': 'boulevard', r'\bimp\b': 'impasse', r'\bpl\b': 'place',
}

LEGAL_MAP = {
    r'\bpvt\s*ltd\b': 'corp', r'\bprivate\s*limited\b': 'corp',
    r'\blimited\b': 'corp', r'\bltd\b': 'corp', r'\bllc\b': 'corp',
    r'\binc\b': 'corp', r'\bincorporated\b': 'corp',
    r'\bcorporation\b': 'corp', r'\bcorp\b': 'corp',
    r'\bco\b': 'corp', r'\bcompany\b': 'corp',
    r'\bsarl\b': 'corp', r'\bgmbh\b': 'corp', r'\bllp\b': 'corp',
    r'\bsa\b': 'corp', r'\bsas\b': 'corp', r'\bsasu\b': 'corp',
    r'\beurl\b': 'corp', r'\bsci\b': 'corp',
}

def fix_leetspeak(t):
    if not isinstance(t, str): return ""
    t = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])|(?<=[a-zA-Z])1\b|\b1(?=[a-zA-Z])', 'l', t)
    t = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])|(?<=[a-zA-Z])0\b|\b0(?=[a-zA-Z])', 'o', t)
    return t

def strip_accents(t):
    if not isinstance(t, str): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')

def normalize_text(t, is_address=False):
    if not isinstance(t, str): return ""
    t = fix_leetspeak(t)
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
    
    # Extract number accurately without matching 'North', 'Nouvelle', or 'Nord'
    num = ''
    pm2 = re.search(r'\b(?:no\.|no\b|plot|door|bldg|flat|h\.?no\.?)\s*[:#-]?\s*(\d+[0-9a-zA-Z\-/]*)', ca, re.I)
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
    
    # Parse address components on original ASCII string before comma-stripping
    pa = parse_addr(addr_ascii)
    nn = normalize_text(name_ascii)
    na = normalize_text(addr_ascii, is_address=True)
    
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
# FEATURE EXTRACTION & EVALUATION
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
        
        ntj = set_jaccard(r1['sig_tokens'], r2['sig_tokens'])
        nts = fuzz.token_sort_ratio(n1, n2) / 100.0
        nlv = fuzz.ratio(n1, n2) / 100.0
        nc3 = set_jaccard(r1['c3_name'], r2['c3_name'])
        
        d1 = s1_wdict.get(s1id, {})
        d2 = c_wdict.get(cid, {})
        twc = sum(v * d2[k] for k, v in d1.items() if k in d2) if d1 and d2 else 0.0
        
        cd1 = s1_cdict.get(s1id, {})
        cd2 = c_cdict.get(cid, {})
        tcc = sum(v * cd2[k] for k, v in cd1.items() if k in cd2) if cd1 and cd2 else 0.0
        
        lex = 1.0 if r1['stripped_name'] and r1['stripped_name'] == r2['stripped_name'] else 0.0
        acr = 1.0 if r1['acronyms'] & r2['acronyms'] else 0.0
        
        if p1['number'] and p2['number']:
            nsm = 1.0 if p1['number'] == p2['number'] else -1.0
            mnm = 0.0
        else:
            nsm = 0.0; mnm = 1.0
            
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
            
        ljc = set_jaccard(p1['loc_tokens'], p2['loc_tokens'])
        st1, st2 = p1['street'], p2['street']
        slv = (fuzz.ratio(st1, st2) / 100.0) if st1 and st2 else 0.0
        ac3 = set_jaccard(r1['c3_addr'], r2['c3_addr'])
        alv = (fuzz.ratio(a1, a2) / 100.0) if a1 and a2 else 0.0
        mad = 1.0 if not a1 or not a2 else 0.0
        
        max_l = max(len(n1), len(n2), 1)
        nld = abs(len(n1) - len(n2)) / max_l
        
        out.append([ntj, nts, nlv, nc3, twc, tcc, lex, acr, nsm, mnm, pcm, mpc, ljc, slv, ac3, alv, mad, nld])
    
    return np.array(out, dtype=np.float32)

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
# MAIN: 3-WAY SPLIT + EVALUATION
# ================================================================
def main():
    T0 = time.time()
    print("=" * 75, flush=True)
    print("  3-WAY SPLIT VALIDATION & FRENCH AUDIT (ZERO LEAKAGE)", flush=True)
    print("=" * 75, flush=True)

    N_TR = 20000
    N_VL = 2500
    N_TE = 2500
    TOT = N_TR + N_VL + N_TE

    # Load S1 & Ground Truth
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
    vl_ids = set(s1ids[N_TR:N_TR + N_VL])
    te_ids = set(s1ids[N_TR + N_VL:])
    
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
                
    print(f"  Splits: Train={len(tr_ids):,} | Val={len(vl_ids):,} | Held-Out Test={len(te_ids):,} ({time.time()-t:.1f}s)", flush=True)

    # Load S2 / S3 Candidates
    t = time.time()
    print(f"\n[2/6] Loading candidates...", flush=True)
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
    print(f"  Candidate pool: {len(cd):,} records ({time.time()-t:.1f}s)", flush=True)

    # Inverted index
    t = time.time()
    print(f"\n[3/6] Building high-recall inverted index...", flush=True)
    inv = {}
    for cid, r in cd.items():
        c = r['country']
        for tok in r['sig_tokens']: inv.setdefault((c, 'T', tok), []).append(cid)
        for num in r['addr_nums']: inv.setdefault((c, 'N', num), []).append(cid)
        for h in r['house_codes']: inv.setdefault((c, 'H', h), []).append(cid)
        for acr in r['acronyms']: inv.setdefault((c, 'A', acr), []).append(cid)
        pref = r['norm_name'][:3]
        if len(pref) >= 3: inv.setdefault((c, 'P', pref), []).append(cid)

    inv_pruned = {k: v for k, v in inv.items() if (len(v) <= 80 if k[1] == 'P' else len(v) <= 300)}
    print(f"  Inverted index built with {len(inv_pruned):,} posting lists ({time.time()-t:.1f}s)", flush=True)

    def query_blocking(r):
        c = r['country']
        cands = set()
        for tok in r['sig_tokens']: cands.update(inv_pruned.get((c, 'T', tok), []))
        for num in r['addr_nums']: cands.update(inv_pruned.get((c, 'N', num), []))
        for h in r['house_codes']: cands.update(inv_pruned.get((c, 'H', h), []))
        for acr in r['acronyms']: cands.update(inv_pruned.get((c, 'A', acr), []))
        pref = r['norm_name'][:3]
        if len(pref) >= 3: cands.update(inv_pruned.get((c, 'P', pref), []))
        return cands

    # Mine hard negatives for training
    t = time.time()
    cands_by_country = {}
    for cid, r in cd.items(): cands_by_country.setdefault(r['country'], []).append(cid)

    train_pairs = []
    for sid in tr_ids:
        r = s1d[sid]
        country = r['country']
        positives = [m for m in gt.get(sid, []) if m in cd]
        for cid in positives: train_pairs.append((sid, cid))
        pos_set = set(positives)
        blocking_cands = query_blocking(r)
        hard_negs = [c for c in blocking_cands if c not in pos_set]
        sampled_hard = random.sample(hard_negs, min(10, len(hard_negs)))
        for cid in sampled_hard:
            train_pairs.append((sid, cid))
            pos_set.add(cid)
        pool = cands_by_country.get(country, [])
        for _ in range(2):
            if pool:
                rc = random.choice(pool)
                if rc not in pos_set:
                    train_pairs.append((sid, rc))
                    pos_set.add(rc)

    print(f"  Train pairs: {len(train_pairs):,} ({time.time()-t:.1f}s)", flush=True)

    # Query Val and Held-Out Test
    val_pairs, val_blocking_stats = [], {}
    for sid in vl_ids:
        r = s1d[sid]; c = r['country']
        cands = query_blocking(r)
        ts = set(gt.get(sid, [])) & set(cd.keys())
        if c not in val_blocking_stats: val_blocking_stats[c] = [0, 0]
        val_blocking_stats[c][0] += len(ts & cands); val_blocking_stats[c][1] += len(ts)
        for cid in cands: val_pairs.append((sid, cid))

    test_pairs, test_blocking_stats = [], {}
    for sid in te_ids:
        r = s1d[sid]; c = r['country']
        cands = query_blocking(r)
        ts = set(gt.get(sid, [])) & set(cd.keys())
        if c not in test_blocking_stats: test_blocking_stats[c] = [0, 0]
        test_blocking_stats[c][0] += len(ts & cands); test_blocking_stats[c][1] += len(ts)
        for cid in cands: test_pairs.append((sid, cid))

    print(f"  Val pairs: {len(val_pairs):,} | Held-Out Test pairs: {len(test_pairs):,}", flush=True)

    # TF-IDF & Fast Dicts
    t = time.time()
    print(f"\n[4/6] Fitting batch TF-IDF & converting to fast dicts...", flush=True)
    all_pairs = train_pairs + val_pairs + test_pairs
    s1_uniq = list({s for s, _ in all_pairs})
    c_uniq = list({c for _, c in all_pairs})
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

    # Extract Train Features & Fit LightGBM
    Xtr = feats_batch(train_pairs, s1d, cd, s1_wdict, c_wdict, s1_cdict, c_cdict)
    ytr = np.array([1 if c in gt.get(s, []) else 0 for s, c in train_pairs], dtype=np.int32)
    print(f"  Xtr extracted: {Xtr.shape} in {time.time()-t:.1f}s", flush=True)

    t_tr = time.time()
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=8,
                         min_child_samples=20, random_state=42, class_weight='balanced',
                         n_jobs=-1, verbose=-1, subsample=0.8, colsample_bytree=0.8)
    clf.fit(Xtr, ytr)
    print(f"  LightGBM trained in {time.time()-t_tr:.1f}s", flush=True)

    # STEP 5: Tune Threshold on VALIDATION Set (Split 2)
    print(f"\n[5/6] Tuning Threshold on Validation Set (Split 2: 2,500 entities)...", flush=True)
    Xvl = feats_batch(val_pairs, s1d, cd, s1_wdict, c_wdict, s1_cdict, c_cdict)
    vp = clf.predict_proba(Xvl)[:, 1]
    vgt = {s: gt.get(s, []) for s in vl_ids}

    best_val_t, best_val_f05 = 0.5, 0.0
    for th in np.arange(0.20, 0.95, 0.05):
        th = round(th, 2)
        cf_preds = collective_resolve(val_pairs, vp, list(vl_ids), pt=th, st=th + 0.10, mm=0.20)
        cf = macro_f05(vgt, cf_preds)
        if cf > best_val_f05:
            best_val_f05 = cf; best_val_t = th
        print(f"    Val T={th:.2f} | Collective F0.5={cf:.4f}", flush=True)

    print(f"  >>> Optimal Threshold chosen on Val: T = {best_val_t:.2f} (Val F0.5 = {best_val_f05:.4f})", flush=True)

    # STEP 6: Strictly Evaluate on Held-Out TEST Set (Split 3) at T_opt ONLY
    print(f"\n[6/6] Strictly Evaluating Held-Out Test Set (Split 3: 2,500 entities) at T = {best_val_t:.2f}...", flush=True)
    Xte = feats_batch(test_pairs, s1d, cd, s1_wdict, c_wdict, s1_cdict, c_cdict)
    tp_scores = clf.predict_proba(Xte)[:, 1]
    tgt = {s: gt.get(s, []) for s in te_ids}

    # Evaluate test at chosen threshold
    test_preds = collective_resolve(test_pairs, tp_scores, list(te_ids), pt=best_val_t, st=best_val_t + 0.10, mm=0.20)
    test_f05 = macro_f05(tgt, test_preds)

    # Per-country test breakdown
    test_country_f05 = {}
    for sid in te_ids:
        c = s1d[sid]['country']
        test_country_f05.setdefault(c, {'gt': {}, 'pred': {}})
        test_country_f05[c]['gt'][sid] = tgt[sid]
        test_country_f05[c]['pred'][sid] = test_preds.get(sid, [])

    print(f"\n  =======================================================", flush=True)
    print(f"  >>> HELD-OUT TEST RESULT (ZERO TUNING LEAKAGE):", flush=True)
    print(f"      Overall Held-Out Test F0.5 = {test_f05:.4f}", flush=True)
    for c in sorted(test_country_f05):
        cf = macro_f05(test_country_f05[c]['gt'], test_country_f05[c]['pred'])
        print(f"      {c:10s} Held-Out Test F0.5 = {cf:.4f}", flush=True)
    print(f"  =======================================================", flush=True)

    # STEP 7: France Open-Set Synthetic Stress-Test
    print(f"\n[7/7] Open-Set French Synthetic Stress-Test Audit...", flush=True)
    french_test_cases = [
        # True matches with perturbations
        ("Team Ecole", "175 Boulevard du President Franklin Roosevelt, Bordeaux",
         "Team Ecole SARL", "175 Bd Roosevelt, Bordeaux, Nouvelle-Aquitaine", 1),
        ("Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France",
         "Thermal & Fils", "20 R. Parmentier, Dunkerque", 1),
        ("Elephant Centre EURL", "30 Rue Lachassaigne, Bordeaux",
         "Elephant Centre", "30 Rue Lachassaigne, Bordeaux, Nouvelle-Aquitaine", 1),
        ("SCI Ptit Amicale", "18 RUE JEN ZAY, Dunkerque, Nord",
         "Ptit Amicale", "18 Rue Jen Zay, Dunkerque", 1),
        ("ZNB Club SARL", "5 bis Rue Pierre Dignac, La Teste-de-Buch",
         "ZNB Club", "5 bis R. Pierre Dignac, La Teste de Buch, Nouvelle-Aquitaine", 1),
        # Non-matches (distinct businesses in same city)
        ("Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE",
         "Tunisie Inter Cie International SARL", "77 AV LEON JOUHAUX, LILLE", 0),
        ("OZT AMICALE SAS", "24 R DESAIX, TOURCOING",
         "Grain & Fils", "329 Avenue de Dunkerque, Lille", 0),
        ("sci ligue ici parents", "NO. 5 ALLEE DES HETRES, Pornic",
         "Team Ecole", "175 Boulevard du President Franklin Roosevelt, Bordeaux", 0),
    ]

    fr_records_s1 = {}
    fr_records_cand = {}
    fr_pairs = []
    fr_labels = []

    for i, (n1, a1, n2, a2, label) in enumerate(french_test_cases):
        s_id = f"FR_S1_{i}"
        c_id = f"FR_C_{i}"
        fr_records_s1[s_id] = make_record(s_id, n1, a1, "France")
        fr_records_cand[c_id] = make_record(c_id, n2, a2, "France")
        fr_pairs.append((s_id, c_id))
        fr_labels.append(label)

    # Compute TF-IDF dicts for French synthetic test
    fr_alln = [r['norm_name'] for r in fr_records_s1.values()] + [r['norm_name'] for r in fr_records_cand.values()]
    fr_s1_names = [r['norm_name'] for r in fr_records_s1.values()]
    fr_c_names = [r['norm_name'] for r in fr_records_cand.values()]
    
    fr_s1_wdict = dict(zip(fr_records_s1.keys(), csr_to_dict_list(wt.transform(fr_s1_names))))
    fr_s1_cdict = dict(zip(fr_records_s1.keys(), csr_to_dict_list(ct.transform(fr_s1_names))))
    fr_c_wdict = dict(zip(fr_records_cand.keys(), csr_to_dict_list(wt.transform(fr_c_names))))
    fr_c_cdict = dict(zip(fr_records_cand.keys(), csr_to_dict_list(ct.transform(fr_c_names))))

    X_fr = feats_batch(fr_pairs, fr_records_s1, fr_records_cand, fr_s1_wdict, fr_c_wdict, fr_s1_cdict, fr_c_cdict)
    fr_probs = clf.predict_proba(X_fr)[:, 1]

    print(f"  French Open-Set Synthetic Evaluation Results:")
    fr_correct = 0
    for (s_id, c_id), p, lab in zip(fr_pairs, fr_probs, fr_labels):
        pred = 1 if p >= best_val_t else 0
        is_ok = (pred == lab)
        if is_ok: fr_correct += 1
        status = "PASS" if is_ok else "FAIL"
        s_rec = fr_records_s1[s_id]
        c_rec = fr_records_cand[c_id]
        print(f"    [{status}] P={p:.4f} (GT={lab}) | S1: {s_rec['norm_name']} @ {s_rec['norm_address'][:30]} vs Cand: {c_rec['norm_name']} @ {c_rec['norm_address'][:30]}", flush=True)

    print(f"\n  French Open-Set Accuracy: {fr_correct}/{len(french_test_cases)} ({fr_correct/len(french_test_cases)*100:.1f}%)", flush=True)
    print(f"\n{'='*75}\n  TOTAL RUNTIME: {time.time()-T0:.1f}s\n{'='*75}", flush=True)

if __name__ == '__main__':
    main()
