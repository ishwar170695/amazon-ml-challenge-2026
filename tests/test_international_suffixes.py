import re
from business_entity_resolution import (
    normalize_text, strip_legal_suffixes, generate_acronyms,
    extract_significant_tokens
)

test_names = [
    ("Siemens AG", "Siemens"),
    ("Unilever PLC", "Unilever Limited"),
    ("Carrefour SA", "Carrefour"),
    ("L'Oréal SAS", "L'Oreal"),
    ("Ferrari N.V.", "Ferrari NV"),
    ("TCS", "Tata Consultancy Services Ltd"),
    ("BMW", "Bayerische Motoren Werke AG"),
]

print("Testing Name Processing on Unrecognized / International Legal Suffixes:\n")
for n1, n2 in test_names:
    norm1, norm2 = normalize_text(n1), normalize_text(n2)
    strip1, strip2 = strip_legal_suffixes(norm1), strip_legal_suffixes(norm2)
    tokens1, tokens2 = extract_significant_tokens(norm1), extract_significant_tokens(norm2)
    acr1, acr2 = generate_acronyms(norm1), generate_acronyms(norm2)
    shared_tokens = tokens1 & tokens2
    shared_acronyms = acr1 & acr2
    
    print(f"• Input: [{n1}] vs [{n2}]")
    print(f"  Normalized: [{norm1}] vs [{norm2}]")
    print(f"  Stripped:   [{strip1}] vs [{strip2}] -> Exact Match: {strip1 == strip2}")
    print(f"  Shared Tokens: {shared_tokens} | Shared Acronyms: {shared_acronyms}\n")
