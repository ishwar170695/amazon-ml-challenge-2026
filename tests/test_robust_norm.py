import unicodedata, re

LEGAL_SUFFIXES_EXT = r'\b(corp|corporation|incorporated|inc|ltd|limited|pvt|private|llc|llp|gmbh|ag|sa|sarl|sas|sasu|plc|bv|nv|spa|srl|sl|cie|co|company)\b'

def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def robust_normalize_text(text: str, is_address: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    text = strip_accents(text.lower().strip())
    # Collapse single-letter dotted initials like n.v. -> nv, s.a. -> sa, m.g. -> mg
    text = re.sub(r'\b([a-z])\.(?:\s*([a-z])\.?)+', lambda m: m.group(0).replace('.', '').replace(' ', ''), text)
    # replace punctuation with space, but preserve words
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def robust_strip_legal(text: str) -> str:
    cleaned = re.sub(LEGAL_SUFFIXES_EXT, '', text.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()

def robust_generate_acronyms(norm_name: str) -> set:
    STOPWORDS = {'and', 'the', '&', 'of', 'in', 'at', 'on', 'for', 'by', 'corp'}
    # Strip legal forms first so BMW AG -> initials of BMW
    cleaned = robust_strip_legal(norm_name)
    alpha_tokens = [t for t in cleaned.split() if t not in STOPWORDS and t.isalpha()]
    acronyms = set()
    if len(alpha_tokens) >= 2:
        acr = ''.join(t[0] for t in alpha_tokens)
        if 2 <= len(acr) <= 6:
            acronyms.add(acr)
    elif len(alpha_tokens) == 1 and 2 <= len(alpha_tokens[0]) <= 5:
        acronyms.add(alpha_tokens[0])
    return acronyms

test_names = [
    ("Siemens AG", "Siemens"),
    ("Unilever PLC", "Unilever Limited"),
    ("Carrefour SA", "Carrefour"),
    ("L'Oréal SAS", "L'Oreal"),
    ("Ferrari N.V.", "Ferrari NV"),
    ("TCS", "Tata Consultancy Services Ltd"),
    ("BMW", "Bayerische Motoren Werke AG"),
    ("Société Générale S.A.", "Societe Generale"),
]

print("Testing Robust International Name Processing:\n")
for n1, n2 in test_names:
    norm1, norm2 = robust_normalize_text(n1), robust_normalize_text(n2)
    strip1, strip2 = robust_strip_legal(norm1), robust_strip_legal(norm2)
    tokens1 = {t for t in norm1.split() if len(t) >= 3}
    tokens2 = {t for t in norm2.split() if len(t) >= 3}
    acr1, acr2 = robust_generate_acronyms(norm1), robust_generate_acronyms(norm2)
    shared_tokens = tokens1 & tokens2
    shared_acronyms = acr1 & acr2
    
    print(f"• Input: [{n1}] vs [{n2}]")
    print(f"  Normalized: [{norm1}] vs [{norm2}]")
    print(f"  Stripped:   [{strip1}] vs [{strip2}] -> Match: {strip1 == strip2}")
    print(f"  Shared Tokens: {shared_tokens} | Shared Acronyms: {shared_acronyms}\n")
