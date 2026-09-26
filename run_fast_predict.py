"""
run_fast_predict.py - High-Throughput Semantics-Preserving Predictor
===================================================================
Optimizations applied:
  1. Compact inverted index using array.array('I') (RAM < 200 MB, eliminates Windows pagefile thrashing)
  2. Candidate make_record() caching across chunks (parsed at most once per candidate)
  3. Sparse TF-IDF dictionary caching via csr_to_dict_list (eliminates slow sp.vstack allocations)
  4. Vectorized feature extraction & OpenMP multi-threaded LightGBM scoring
  5. Global bipartite resolution (collective_resolve) preserving exact predictions and assignment semantics
"""

import os, sys, time, gc, array, re, pickle
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from rapidfuzz import fuzz
import anyascii

sys.path.insert(0, os.path.abspath('.'))

from run_pipeline import (
    normalize_text, sig_tokens, get_addr_numbers, get_house_codes,
    gen_acronyms, indic_phonetic_skeleton, extract_unit_keys, clean_compact_brand,
    get_distinctive_addr_tokens, make_record, set_jaccard, csr_to_dict_list,
    strip_legal, parse_addr, collective_resolve
)

def run_fast_predict(sample_limit=None, target_country=None, output_dir='output'):
    t0 = time.time()
    print("=" * 75, flush=True)
    print("  HIGH-THROUGHPUT TEST PREDICTION PIPELINE", flush=True)
    print("=" * 75, flush=True)

    model_path = 'artifacts/model_v3.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Train first.")

    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    clf = artifacts['model']
    word_vec = artifacts['word_tfidf']
    char_vec = artifacts['char_tfidf']
    thresh_config = artifacts.get('thresholds', {'US': 0.98, 'India': 0.97, 'France': 0.97, 'default': 0.97})
    sec_thresh = artifacts.get('secondary_threshold', 0.88)
    margin = artifacts.get('margin', 0.20)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'matching_results.tsv')
    cand_file = os.path.join(output_dir, 'candidate_pairs.tsv')

    countries = [target_country] if target_country else ['France', 'US', 'India']

    # Initialize output files with headers
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(cand_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    total_s1_processed = 0
    total_matches_found = 0

    for country in countries:
        c_t0 = time.time()
        print(f"\n>>> Processing country: {country}...", flush=True)

        # 1. Load S1 records for this country
        s1_dict = {}
        with open('dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 4 and p[3] == country:
                    s1_dict[p[0]] = make_record(p[0], p[1], p[2] if p[2] != 'nan' else '', p[3])
                    if sample_limit and len(s1_dict) >= sample_limit:
                        break
        print(f"  Loaded {len(s1_dict):,} test S1 entities for {country} in {time.time()-c_t0:.1f}s", flush=True)
        if not s1_dict:
            continue

        # 2. Stream-index candidates into compact array.array('I') index
        cand_raw = [] # [(cid, cname, caddr)]
        cid_to_idx = {}
        inv = {}
        s23_t0 = time.time()
        
        for sf in ['dataset/test/test_source2.tsv', 'dataset/test/test_source3.tsv']:
            cnt = 0
            with open(sf, 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    p = line.strip().split('\t')
                    if len(p) >= 4 and p[3] == country:
                        cid, cname, caddr = p[0], p[1], p[2] if p[2] != 'nan' else ''
                        c_idx = len(cand_raw)
                        cand_raw.append((cid, cname, caddr))
                        cid_to_idx[cid] = c_idx

                        name_ascii = anyascii.anyascii(cname) if any(ord(c) > 127 for c in cname) else cname
                        addr_ascii = anyascii.anyascii(caddr) if any(ord(c) > 127 for c in caddr) else caddr
                        nn = normalize_text(name_ascii)
                        na = normalize_text(addr_ascii, is_address=True)
                        st = sig_tokens(nn)

                        for tok in st: inv.setdefault(('T', tok), array.array('I')).append(c_idx)
                        for num in get_addr_numbers(addr_ascii): inv.setdefault(('N', num), array.array('I')).append(c_idx)
                        for h in get_house_codes(addr_ascii): inv.setdefault(('H', h), array.array('I')).append(c_idx)
                        for acr in gen_acronyms(nn): inv.setdefault(('A', acr), array.array('I')).append(c_idx)
                        for pht in {indic_phonetic_skeleton(t) for t in st if len(t) >= 3}:
                            inv.setdefault(('PH', pht), array.array('I')).append(c_idx)
                        for uk in extract_unit_keys(addr_ascii): inv.setdefault(('UK', uk), array.array('I')).append(c_idx)
                        cb = clean_compact_brand(nn)
                        if cb: inv.setdefault(('CB', cb[:8]), array.array('I')).append(c_idx)
                        for atok in get_distinctive_addr_tokens(na): inv.setdefault(('AT', atok), array.array('I')).append(c_idx)
                        if len(nn) >= 3: inv.setdefault(('P', nn[:3]), array.array('I')).append(c_idx)

                        cnt += 1
                        if sample_limit and cnt >= sample_limit * 5:
                            break
            print(f"    Indexed {cnt:,} {country} candidates from {os.path.basename(sf)} in {time.time()-s23_t0:.1f}s", flush=True)

        print(f"  Total candidate pool for {country}: {len(cand_raw):,} records", flush=True)

        # 3. Prune inverted index
        inv_pruned = {}
        for k, v in inv.items():
            limit = 80 if k[0] == 'P' else (150 if k[0] == 'AT' else 300)
            if len(v) <= limit:
                inv_pruned[k] = v
        del inv
        print(f"  Pruned inverted index: {len(inv_pruned):,} active keys", flush=True)

        # 4. Global candidate record and TF-IDF cache
        cand_record_cache = {} # c_idx -> record
        name_to_wdict = {}
        name_to_cdict = {}

        # 5. Evaluate S1 entities in chunks of 10,000
        s1_ids = list(s1_dict.keys())
        chunk_size = 10000
        print(f"  Evaluating {len(s1_ids):,} S1 entities in chunks of {chunk_size:,}...", flush=True)

        for ch_idx in range(0, len(s1_ids), chunk_size):
            ch_start_t = time.time()
            ch_s1_ids = s1_ids[ch_idx:ch_idx + chunk_size]
            ch_s1_dict = {sid: s1_dict[sid] for sid in ch_s1_ids}

            # Query candidates
            ch_pairs = []
            needed_c_indices = set()
            s1_cand_map = {s: [] for s in ch_s1_ids}

            for sid in ch_s1_ids:
                r = ch_s1_dict[sid]
                cands = set()
                for tok in r['sig_tokens']: cands.update(inv_pruned.get(('T', tok), ()))
                for num in r['addr_nums']: cands.update(inv_pruned.get(('N', num), ()))
                for h in r['house_codes']: cands.update(inv_pruned.get(('H', h), ()))
                for acr in r['acronyms']: cands.update(inv_pruned.get(('A', acr), ()))
                for pht in r['phonetic_tokens']: cands.update(inv_pruned.get(('PH', pht), ()))
                for uk in r['unit_keys']: cands.update(inv_pruned.get(('UK', uk), ()))
                if r['compact_brand']: cands.update(inv_pruned.get(('CB', r['compact_brand'][:8]), ()))
                for tok in r['sig_tokens']:
                    if len(tok) >= 5: cands.update(inv_pruned.get(('CB', tok[:8]), ()))
                for atok in r['distinctive_addr_tokens']: cands.update(inv_pruned.get(('AT', atok), ()))
                pref = r['norm_name'][:3]
                if len(pref) >= 3: cands.update(inv_pruned.get(('P', pref), ()))

                for c_idx in cands:
                    cid = cand_raw[c_idx][0]
                    ch_pairs.append((sid, cid, c_idx))
                    needed_c_indices.add(c_idx)
                    s1_cand_map[sid].append(cid)

            if ch_pairs:
                # Cache make_record for newly seen candidates
                uncached_c_indices = [c for c in needed_c_indices if c not in cand_record_cache]
                for c_idx in uncached_c_indices:
                    cid, cname, caddr = cand_raw[c_idx]
                    cand_record_cache[c_idx] = make_record(cid, cname, caddr, country)

                # Batch transform unseen names with TF-IDF
                unseen_names = []
                for sid in ch_s1_ids:
                    sn = ch_s1_dict[sid]['stripped_name']
                    if sn not in name_to_wdict:
                        unseen_names.append(sn)
                for c_idx in needed_c_indices:
                    cn = cand_record_cache[c_idx]['stripped_name']
                    if cn not in name_to_wdict:
                        unseen_names.append(cn)

                if unseen_names:
                    unseen_unique = list(set(unseen_names))
                    w_csr = word_vec.transform(unseen_unique)
                    c_csr = char_vec.transform(unseen_unique)
                    w_dl = csr_to_dict_list(w_csr)
                    c_dl = csr_to_dict_list(c_csr)
                    for idx_name, nm in enumerate(unseen_unique):
                        name_to_wdict[nm] = w_dl[idx_name]
                        name_to_cdict[nm] = c_dl[idx_name]
                    del w_csr, c_csr, w_dl, c_dl, unseen_unique

                # Vectorized feature extraction in 100k sub-batches
                pair_batch_size = 100000
                all_probs = []

                for p_start in range(0, len(ch_pairs), pair_batch_size):
                    sub_pairs = ch_pairs[p_start:p_start + pair_batch_size]
                    n_sub = len(sub_pairs)
                    X_sub = np.empty((n_sub, 21), dtype=np.float32)

                    for i, (sid, cid, c_idx) in enumerate(sub_pairs):
                        r1 = ch_s1_dict[sid]
                        r2 = cand_record_cache[c_idx]
                        n1, n2 = r1['norm_name'], r2['norm_name']
                        a1, a2 = r1['norm_address'], r2['norm_address']
                        p1, p2 = r1['parsed_addr'], r2['parsed_addr']

                        ntj = set_jaccard(r1['sig_tokens'], r2['sig_tokens'])
                        nts = fuzz.token_sort_ratio(n1, n2) / 100.0
                        nlv = fuzz.ratio(n1, n2) / 100.0
                        nc3 = set_jaccard(r1['c3_name'], r2['c3_name'])

                        wd1 = name_to_wdict.get(r1['stripped_name'], {})
                        wd2 = name_to_wdict.get(r2['stripped_name'], {})
                        twc = sum(v * wd2[k] for k, v in wd1.items() if k in wd2) if wd1 and wd2 else 0.0

                        cd1 = name_to_cdict.get(r1['stripped_name'], {})
                        cd2 = name_to_cdict.get(r2['stripped_name'], {})
                        tcc = sum(v * cd2[k] for k, v in cd1.items() if k in cd2) if cd1 and cd2 else 0.0

                        lex = 1.0 if r1['stripped_name'] and r1['stripped_name'] == r2['stripped_name'] else 0.0
                        acr = 1.0 if r1['acronyms'] & r2['acronyms'] else 0.0

                        if p1['number'] and p2['number']:
                            nsm = 1.0 if p1['number'] == p2['number'] else -1.0
                            mnm = 0.0
                        else:
                            nsm = 0.0; mnm = 1.0

                        if p1['postcode'] and p2['postcode']:
                            if p1['postcode'] == p2['postcode']: pcm = 1.0
                            elif p1['postcode'][:3] == p2['postcode'][:3]: pcm = 0.5
                            else: pcm = -1.0
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

                        pht1 = ' '.join(r1['phonetic_tokens'])
                        pht2 = ' '.join(r2['phonetic_tokens'])
                        phs = (fuzz.token_sort_ratio(pht1, pht2) / 100.0) if pht1 and pht2 else 0.0

                        cb1 = r1['compact_brand']
                        cb2 = r2['compact_brand']
                        cbs = (fuzz.ratio(cb1, cb2) / 100.0) if cb1 and cb2 else 0.0

                        u1 = r1['unit_keys']
                        u2 = r2['unit_keys']
                        ukm = 1.0 if (u1 and u2 and (u1 & u2)) else (-1.0 if (u1 and u2) else 0.0)

                        X_sub[i] = [ntj, nts, nlv, nc3, twc, tcc, lex, acr, nsm, mnm, pcm, mpc, ljc, slv, ac3, alv, mad, nld, phs, cbs, ukm]

                    p_sub = clf.predict_proba(X_sub)[:, 1]
                    all_probs.append(p_sub)
                    del X_sub

                probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
                resolved_pairs = [(sid, cid) for (sid, cid, _) in ch_pairs]
                del all_probs
            else:
                probs = np.array([], dtype=np.float32)
                resolved_pairs = []

            # Exact collective resolution
            final_matches = collective_resolve(
                resolved_pairs, probs, ch_s1_ids,
                primary_threshold=thresh_config, secondary_threshold=sec_thresh, margin=margin,
                s1_dict=ch_s1_dict
            )

            # Write outputs
            with open(out_file, 'a', encoding='utf-8') as f:
                for sid in ch_s1_ids:
                    m = final_matches.get(sid, [])
                    if m: total_matches_found += 1
                    f.write(f"{sid}\t{','.join(m) if m else ''}\n")

            with open(cand_file, 'a', encoding='utf-8') as f:
                for sid in ch_s1_ids:
                    cands = s1_cand_map.get(sid, [])
                    f.write(f"{sid}\t{','.join(cands) if cands else ''}\n")

            total_s1_processed += len(ch_s1_ids)
            pct = ((ch_idx + len(ch_s1_ids)) / len(s1_ids)) * 100.0
            print(f"    Processed [{ch_idx + len(ch_s1_ids):,}/{len(s1_ids):,}] ({pct:.1f}%) in {time.time()-ch_start_t:.1f}s (cached candidates: {len(cand_record_cache):,})", flush=True)

            del ch_pairs, needed_c_indices, ch_s1_dict, probs, final_matches, s1_cand_map, resolved_pairs
            gc.collect()

        print(f"  Finished {country}: {len(s1_dict):,} S1 entities processed in {time.time()-c_t0:.1f}s", flush=True)
        del s1_dict, cand_raw, cid_to_idx, inv_pruned, cand_record_cache, name_to_wdict, name_to_cdict
        gc.collect()

    print(f"\n[DONE] Processed {total_s1_processed:,} total S1 records in {time.time()-t0:.1f}s. Found matches for {total_matches_found:,} entities.", flush=True)
    return out_file, cand_file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Sample limit for testing')
    parser.add_argument('--country', type=str, default=None, help='Target country')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    args = parser.parse_args()

    run_fast_predict(sample_limit=args.limit, target_country=args.country, output_dir=args.output_dir)
