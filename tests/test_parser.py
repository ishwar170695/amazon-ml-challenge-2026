import re

def robust_parse_address_components(addr: str) -> dict:
    if not isinstance(addr, str) or not addr.strip():
        return {'number': '', 'postcode': '', 'street': '', 'locality': ''}
    addr = addr.strip()

    # 1. Postcode / PIN (5-6 digits)
    p_match = re.search(r'\b\d{5,6}\b', addr)
    postcode = p_match.group(0) if p_match else ''
    clean_addr = re.sub(r'\b\d{5,6}\b', '', addr).strip()

    number = ''
    # Pattern A: Prefixed by 'no.', 'plot', 'door', 'bldg'
    pref_match = re.search(r'\b(?:no\.?|plot|door|bldg)\s*[:#-]?\s*(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', clean_addr, re.IGNORECASE)
    if pref_match:
        number = pref_match.group(1).strip().lower()
        clean_addr = clean_addr[:pref_match.start()] + clean_addr[pref_match.end():]
    else:
        # Pattern B: Leading number (e.g. '105 Main St', '14 bis rue...')
        lead_match = re.match(r'^(\d+\s*(?:bis|ter|[a-zA-Z])?)\b', clean_addr, re.IGNORECASE)
        if lead_match:
            number = lead_match.group(1).strip().lower()
            clean_addr = clean_addr[lead_match.end():].strip(', ')
        else:
            # Pattern C: Trailing number on street phrase (European format e.g. 'Rue de la Paix 14', 'R. de la Paix 188')
            parts_temp = [p.strip() for p in clean_addr.split(',') if p.strip()]
            if parts_temp:
                street_cand = parts_temp[0]
                trail_match = re.search(r'\b(\d+\s*(?:bis|ter|[a-zA-Z])?)\s*$', street_cand, re.IGNORECASE)
                if trail_match:
                    number = trail_match.group(1).strip().lower()
                    parts_temp[0] = street_cand[:trail_match.start()].strip()
                    clean_addr = ', '.join(parts_temp)

    # 3. Locality vs Street
    parts = [p.strip() for p in clean_addr.split(',') if p.strip()]
    locality = parts[-1].lower() if len(parts) > 1 else ''
    street = ', '.join(parts[:-1]).lower() if len(parts) > 1 else (parts[0].lower() if parts else '')

    return {
        'number': number,
        'postcode': postcode,
        'street': street,
        'locality': locality
    }

print("S1:", robust_parse_address_components('Rue de la Paix 188, Bordeaux'))
print("Cand:", robust_parse_address_components('R. de la Paix 188'))
