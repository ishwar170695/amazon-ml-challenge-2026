"""
Full-Featured Real-Data Pipeline v2.2
KEY FIX: Training pairs built directly from GT + random negatives (no blocking loop).
         Validation blocking done via forward-index build in single pass.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, re, time, random, pickle, unicodedata
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib

random.seed(42); np.random.seed(42)

# ================================================================
# TEXT NORMALIZATION (Full)
# ================================================================
LEGAL_SUFFIXES_REGEX = r'\b(corp|corporation|incorporated|inc|ltd|limited|pvt|private|llc|llp|gmbh|ag|sa|sarl|sas|sasu|plc|bv|nv|spa|srl|sl|cie|co|company)\b'
STOPWORDS = {'and', 'the', '&', 'of', 'in', 'at', 'on', 'for', 'by', 'corp'}
ADDRESS_ABBREVS = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bln\b': 'lane', r'\bbldg\b': 'building',
    r'\bapt\b': 'apartment', r'\bste\b': 'suite', r'\bfl\b': 'floor',
    r'\bdr\b': 'drive', r'\bct\b': 'court', r'\bpkwy\b': 'parkway',
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

def sig_tokens(nn): return {t for t in nn.split() if len(t)>=3 and t not in STOPWORDS}
def strip_legal(n): return re.sub(r'\s+',' ',re.sub(LEGAL_SUFFIXES_REGEX,'',n.lower())).strip()
def gen_acronyms(nn):
    cl = strip_legal(nn)
    at = [t for t in cl.split() if t not in STOPWORDS and t.isalpha()]
    acrs = set()
    if len(at)>=2:
        a=''.join(t[0] for t in at)
        if 2<=len(a)<=6: acrs.add(a)
    elif len(at)==1 and 2<=len(at[0])<=5: acrs.add(at[0])
    return acrs

def parse_addr(addr):
    if not isinstance(addr, str) or not addr.strip():
        return {'number':'','postcode':'','street':'','locality':''}
    a = addr.strip()
    pm = re.search(r'\b\d{5,6}\b', a)
    pc = pm.group(0) if pm else ''
    ca = re.sub(r'\b\d{5,6}\b','',a).strip()
    num = ''
    pm2 = re.search(r'\b(?:no\.?|plot|door|bldg)\s*[:#-]?\s*(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', ca, re.I)
    if pm2: num=pm2.group(1).strip().lower(); ca=ca[:pm2.start()]+ca[pm2.end():]
    else:
        lm = re.match(r'^(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', ca, re.I)
        if lm: num=lm.group(1).strip().lower(); ca=ca[lm.end():].strip(', ')
        else:
            pts=[p.strip() for p in ca.split(',') if p.strip()]
            if pts:
                tm=re.search(r'\b(\d+\s*(?:bis|ter|[a-zA-Z])?)\s*$',pts[0],re.I)
                if tm: num=tm.group(1).strip().lower(); pts[0]=pts[0][:tm.start()].strip(); ca=', '.join(pts)
    pts=[p.strip() for p in ca.split(',') if p.strip()]
    loc = pts[-1].lower() if len(pts)>1 else ''
    st = ', '.join(pts[:-1]).lower() if len(pts)>1 else (pts[0].lower() if pts else '')
    return {'number':num,'postcode':pc,'street':st,'locality':loc}

def dist_nums(t):
    if not isinstance(t, str): return set()
    return set(re.findall(r'\b\d{3,6}\b', t))

def make_record(eid, name, addr, country):
    nn = normalize_text(name)
    na = normalize_text(addr, is_address=True)
    return {
        'entity_id':eid, 'business_name':name, 'business_address':addr, 'country':country,
        'norm_name':nn, 'norm_address':na, 'sig_tokens':sig_tokens(nn),
        'acronyms':gen_acronyms(nn), 'parsed_addr':parse_addr(addr),
        'stripped_name':strip_legal(nn), 'addr_nums':dist_nums(addr),
    }

# ================================================================
# SIMILARITY + FEATURES
# ================================================================
def jaccard(a,b):
    if not a or not b: return 0.0
    u=len(a|b); return len(a&b)/u if u else 0.0
def lev(a,b):
    if not a or not b: return 0.0
    if a==b: return 1.0
    return difflib.SequenceMatcher(None,a,b).ratio()
def tok_sort(a,b): return lev(' '.join(sorted(a.split())),' '.join(sorted(b.split())))
def cng(a,b,n=3):
    if len(a)<n or len(b)<n: return 0.0
    s1={a[i:i+n] for i in range(len(a)-n+1)}; s2={b[i:i+n] for i in range(len(b)-n+1)}
    u=s1|s2; return len(s1&s2)/len(u) if u else 0.0

FCOLS = ['ntj','nts','nlv','nc3','twc','tcc','lex','acr','nsm','mnm','pcm','mpc','ljc','slv','ac3','nld']

def feats_batch(pairs, s1d, cd, s1wv, cwv, s1cv, ccv, s1i, ci):
    out = []
    for s1id,cid in pairs:
        r1,r2 = s1d[s1id], cd[cid]
        n1,n2 = r1['norm_name'],r2['norm_name']
        a1,a2 = r1['norm_address'],r2['norm_address']
        p1,p2 = r1['parsed_addr'],r2['parsed_addr']
        # TF-IDF cosines via pre-computed sparse dot
        tw = float(s1wv[s1i[s1id]].dot(cwv[ci[cid]].T).toarray()[0,0])
        tc = float(s1cv[s1i[s1id]].dot(ccv[ci[cid]].T).toarray()[0,0])
        # Number
        if p1['number'] and p2['number']: ns=1.0 if p1['number']==p2['number'] else -1.0; mn=0.0
        else: ns=0.0; mn=1.0
        # PIN
        if p1['postcode'] and p2['postcode']:
            ps=1.0 if p1['postcode']==p2['postcode'] else (0.5 if p1['postcode'][:3]==p2['postcode'][:3] else -1.0); mp=0.0
        else: ps=0.0; mp=1.0

        out.append([
            jaccard(r1['sig_tokens'],r2['sig_tokens']),
            tok_sort(n1,n2), lev(n1,n2), cng(n1,n2,3), tw, tc,
            1.0 if r1['stripped_name'] and r1['stripped_name']==r2['stripped_name'] else 0.0,
            1.0 if r1['acronyms']&r2['acronyms'] else 0.0,
            ns, mn, ps, mp,
            jaccard(set(p1['locality'].split()),set(p2['locality'].split())),
            lev(p1['street'],p2['street']), cng(a1,a2,3),
            abs(len(n1)-len(n2))/max(len(n1),len(n2),1)
        ])
    return np.array(out, dtype=np.float32)

# ================================================================
# METRICS + COLLECTIVE RESOLVE
# ================================================================
def macro_f05(gt, pred):
    scores=[]
    for s1,tr in gt.items():
        ts=set(tr); ps=set(pred.get(s1,[]))
        if not ts: scores.append(1.0 if not ps else 0.0)
        elif not ps: scores.append(0.0)
        else:
            tp=len(ts&ps); p=tp/len(ps); r=tp/len(ts)
            scores.append((1.25*p*r)/(0.25*p+r) if p+r>0 else 0.0)
    return float(np.mean(scores))

def collective_resolve(pairs, probs, eids, pt=0.4, st=0.55, mm=0.25):
    s1c={s:[] for s in eids}; cc={}
    for (s1,c),p in zip(pairs,probs):
        if s1 in s1c and p>=pt:
            s1c[s1].append((c,p)); cc.setdefault(c,[]).append((p,s1))
    cw={}
    for c,cl in cc.items(): cl.sort(reverse=True,key=lambda x:x[0]); cw[c]=cl[0][1]
    fm={s:[] for s in eids}
    for s1,cds in s1c.items():
        if not cds: continue
        cds.sort(reverse=True,key=lambda x:x[1])
        tc,tp=cds[0]
        if cw.get(tc)==s1: fm[s1].append(tc)
        for c,p in cds[1:]:
            if cw.get(c)==s1 and p>=st and (tp-p)<=mm: fm[s1].append(c)
    return fm

# ================================================================
# MAIN
# ================================================================
def main():
    T0 = time.time()
    print("="*70, flush=True)
    print("  PIPELINE v2.2: GT-Direct Training + Forward-Index Validation", flush=True)
    print("="*70, flush=True)

    N_TR, N_VL = 20000, 5000
    TOT = N_TR + N_VL

    # STEP 1: Load S1 + GT
    t=time.time()
    print(f"\n[1/6] Loading {TOT:,} S1 + GT...", flush=True)
    s1d = {}
    with open('dataset/train/train_source1.tsv','r',encoding='utf-8') as f:
        f.readline()
        for i,line in enumerate(f):
            p=line.strip().split('\t')
            if len(p)>=4:
                s1d[p[0]] = make_record(p[0],p[1],p[2] if p[2]!='nan' else '',p[3])
            if len(s1d)>=TOT: break
    s1ids=list(s1d.keys()); tr_ids=set(s1ids[:N_TR]); vl_ids=set(s1ids[N_TR:])
    gt={}; needed=set()
    with open('dataset/train/train_ground_truth.tsv','r',encoding='utf-8') as f:
        f.readline()
        for line in f:
            p=line.strip().split('\t')
            if p and p[0] in s1d:
                m=[x.strip() for x in p[1].split(',') if x.strip() and x.strip()!='nan'] if len(p)>1 else []
                gt[p[0]]=m; needed.update(m)
    print(f"  Done: {len(s1d):,} S1 | {len(needed):,} targets ({time.time()-t:.1f}s)", flush=True)

    # STEP 2: Load candidates
    t=time.time()
    print(f"\n[2/6] Loading candidates...", flush=True)
    cd = {}
    for sf, rem in [('dataset/train/train_source2.tsv',{x for x in needed if x.startswith('S2-')}),
                     ('dataset/train/train_source3.tsv',{x for x in needed if x.startswith('S3-')})]:
        bg=0
        with open(sf,'r',encoding='utf-8') as f:
            f.readline()
            for line in f:
                p=line.strip().split('\t')
                if len(p)>=4:
                    cid=p[0]; isn=cid in rem
                    if isn or bg<15000:
                        cd[cid]=make_record(cid,p[1],p[2] if p[2]!='nan' else '',p[3])
                        if isn: rem.discard(cid)
                        else: bg+=1
                    if not rem and bg>=15000: break
    print(f"  Candidates: {len(cd):,} ({time.time()-t:.1f}s)", flush=True)

    # STEP 3: Build training pairs DIRECTLY from GT (no blocking loop!)
    t=time.time()
    print(f"\n[3/6] Building training pairs from GT + random negatives...", flush=True)
    # Group candidates by country for random negative sampling
    cands_by_country = {}
    for cid, r in cd.items():
        cands_by_country.setdefault(r['country'], []).append(cid)

    train_pairs = []
    for sid in tr_ids:
        country = s1d[sid]['country']
        positives = [m for m in gt.get(sid, []) if m in cd]
        for cid in positives:
            train_pairs.append((sid, cid))
        # Sample hard negatives from same country
        pool = cands_by_country.get(country, [])
        pos_set = set(positives)
        neg_sample = []
        attempts = 0
        while len(neg_sample) < min(15, len(pool)) and attempts < 100:
            c = random.choice(pool)
            if c not in pos_set:
                neg_sample.append(c)
                pos_set.add(c)  # avoid dups
            attempts += 1
        for cid in neg_sample:
            train_pairs.append((sid, cid))

    print(f"  Train pairs: {len(train_pairs):,} ({time.time()-t:.1f}s)", flush=True)

    # Build validation blocking via forward index (single pass)
    t=time.time()
    print(f"  Building validation forward-index blocking...", flush=True)
    # Build inverted index ONLY over candidates
    inv = {}  # (country, key) -> [cid]
    for cid, r in cd.items():
        c = r['country']
        for tok in r['sig_tokens']:
            inv.setdefault((c, 'T', tok), []).append(cid)
        for num in r['addr_nums']:
            inv.setdefault((c, 'N', num), []).append(cid)
        for acr in r['acronyms']:
            inv.setdefault((c, 'A', acr), []).append(cid)
        pref = r['norm_name'][:3]
        if len(pref) >= 3:
            inv.setdefault((c, 'P', pref), []).append(cid)

    # Prune
    inv = {k: v for k, v in inv.items() if len(v) <= 500}

    # Query for val entities only (5000 entities — fast)
    val_pairs = []
    val_blocking_stats = {}
    for sid in vl_ids:
        r = s1d[sid]
        c = r['country']
        cands = set()
        for tok in r['sig_tokens']:
            cands.update(inv.get((c,'T',tok), []))
        for num in r['addr_nums']:
            cands.update(inv.get((c,'N',num), []))
        for acr in r['acronyms']:
            cands.update(inv.get((c,'A',acr), []))
        pref = r['norm_name'][:3]
        if len(pref) >= 3:
            cands.update(inv.get((c,'P',pref), []))

        true_set = set(gt.get(sid,[])) & set(cd.keys())
        cap = len(true_set & cands)
        tot = len(true_set)
        if c not in val_blocking_stats: val_blocking_stats[c] = [0,0]
        val_blocking_stats[c][0] += cap
        val_blocking_stats[c][1] += tot

        for cid in cands:
            val_pairs.append((sid, cid))

    print(f"\n  --- Per-Country Blocking Recall (UNCAPPED, val only) ---", flush=True)
    tc,tt = 0,0
    for c in sorted(val_blocking_stats):
        cap,tot = val_blocking_stats[c]
        tc+=cap; tt+=tot
        print(f"    {c:8s}: {cap}/{tot} ({cap/tot*100:.2f}%)" if tot else f"    {c:8s}: 0/0", flush=True)
    print(f"    {'OVERALL':8s}: {tc}/{tt} ({tc/tt*100:.2f}%)", flush=True)
    print(f"  Val pairs: {len(val_pairs):,} ({time.time()-t:.1f}s)", flush=True)

    # STEP 4: Batch TF-IDF + features
    t=time.time()
    print(f"\n[4/6] Batch TF-IDF + 16-feature extraction...", flush=True)
    s1_uniq = list({s for s,_ in train_pairs + val_pairs})
    c_uniq = list({c for _,c in train_pairs + val_pairs})
    s1n = [s1d[s]['norm_name'] for s in s1_uniq]
    cn = [cd[c]['norm_name'] for c in c_uniq]
    alln = s1n + cn
    print(f"  Fitting TF-IDF on {len(alln):,} names...", flush=True)
    wt = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.95).fit(alln)
    ct = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,min_df=2,max_df=0.95).fit(alln)
    del alln
    s1wv=wt.transform(s1n); s1cv=ct.transform(s1n); s1i={s:i for i,s in enumerate(s1_uniq)}
    cwv=wt.transform(cn); ccv=ct.transform(cn); ci={c:i for i,c in enumerate(c_uniq)}
    del s1n, cn
    print(f"  TF-IDF done in {time.time()-t:.1f}s (Word: {len(wt.vocabulary_):,}, Char: {len(ct.vocabulary_):,})", flush=True)

    t=time.time()
    print(f"  Extracting {len(train_pairs):,} train features...", flush=True)
    Xtr = feats_batch(train_pairs, s1d, cd, s1wv, cwv, s1cv, ccv, s1i, ci)
    ytr = np.array([1 if c in gt.get(s,[]) else 0 for s,c in train_pairs], dtype=np.int32)
    print(f"  Xtr: {Xtr.shape} | Pos: {np.sum(ytr):,} ({np.mean(ytr)*100:.1f}%) ({time.time()-t:.1f}s)", flush=True)

    # STEP 5: Train
    t=time.time()
    print(f"\n[5/6] Training LightGBM...", flush=True)
    clf = LGBMClassifier(n_estimators=300,learning_rate=0.06,num_leaves=63,max_depth=8,
                          min_child_samples=20,random_state=42,class_weight='balanced',
                          n_jobs=-1,verbose=-1,subsample=0.8,colsample_bytree=0.8)
    clf.fit(Xtr, ytr)
    print(f"  Trained in {time.time()-t:.1f}s!", flush=True)
    os.makedirs('artifacts',exist_ok=True)
    with open('artifacts/model_v2.pkl','wb') as f:
        pickle.dump({'model':clf,'word_tfidf':wt,'char_tfidf':ct},f)

    print(f"\n  --- Feature Importances ---", flush=True)
    imp = pd.Series(clf.feature_importances_, index=FCOLS).sort_values(ascending=False)
    for c,v in imp.items(): print(f"    {c:6s}: {v:5d} ({v/imp.sum()*100:5.1f}%)", flush=True)

    # STEP 6: Validation
    t=time.time()
    print(f"\n[6/6] Validation + threshold sweep + collective...", flush=True)
    Xvl = feats_batch(val_pairs, s1d, cd, s1wv, cwv, s1cv, ccv, s1i, ci)
    vp = clf.predict_proba(Xvl)[:,1]
    print(f"  Val features in {time.time()-t:.1f}s", flush=True)

    vgt = {s: gt.get(s,[]) for s in vl_ids}

    print(f"\n  --- Threshold Sweep ---", flush=True)
    bt,bf,bct,bcf = 0.5,0.0,0.5,0.0
    for th in np.arange(0.30,0.91,0.05):
        th=round(th,2)
        pw={s:[] for s in vl_ids}
        for (s,c),p in zip(val_pairs,vp):
            if p>=th: pw[s].append(c)
        pf = macro_f05(vgt, pw)
        cf_preds = collective_resolve(val_pairs,vp,list(vl_ids),pt=th,st=th+0.10,mm=0.25)
        cf = macro_f05(vgt, cf_preds)
        m=""
        if pf>bf: bf=pf;bt=th
        if cf>bcf: bcf=cf;bct=th;m=" <--"
        print(f"    T={th:.2f} | PW={pf:.4f} | Coll={cf:.4f}{m}", flush=True)

    print(f"\n  Best PW: T={bt:.2f} F0.5={bf:.4f}", flush=True)
    print(f"  Best Coll: T={bct:.2f} F0.5={bcf:.4f}", flush=True)

    # Error analysis
    bp = collective_resolve(val_pairs,vp,list(vl_ids),pt=bct,st=bct+0.10,mm=0.25)
    fps,fns=[],[]
    for s in vl_ids:
        ts=set(vgt[s]); ps=set(bp.get(s,[]))
        for c in ps-ts: fps.append((s,c))
        for c in ts-ps:
            if c in cd: fns.append((s,c))
    print(f"\n  FPs: {len(fps)} | FNs: {len(fns)}", flush=True)
    if fps:
        print(f"  Sample FPs:", flush=True)
        for s,c in fps[:5]:
            print(f"    S1: {s1d[s]['business_name'][:50]} @ {str(s1d[s]['business_address'])[:35]}", flush=True)
            print(f"    FP: {cd[c]['business_name'][:50]} @ {str(cd[c]['business_address'])[:35]}", flush=True)
    if fns:
        print(f"  Sample FNs:", flush=True)
        for s,c in fns[:5]:
            print(f"    S1: {s1d[s]['business_name'][:50]} @ {str(s1d[s]['business_address'])[:35]}", flush=True)
            print(f"    FN: {cd[c]['business_name'][:50]} @ {str(cd[c]['business_address'])[:35]}", flush=True)

    print(f"\n{'='*70}\n  Total: {time.time()-T0:.1f}s\n{'='*70}", flush=True)
    import json
    with open('artifacts/result_v2.json','w') as f:
        json.dump({'pw_t':bt,'pw_f05':bf,'coll_t':bct,'coll_f05':bcf,
                   'fps':len(fps),'fns':len(fns),
                   'blocking':{c:[s[0],s[1]] for c,s in val_blocking_stats.items()}},f,indent=2)

if __name__=='__main__': main()
