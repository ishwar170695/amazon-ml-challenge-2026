import re

def clean_dotted_acronyms(text: str) -> str:
    # Collapse dotted abbreviations like N.V. -> NV, S.A. -> SA, U.S.A. -> USA
    # Repeatedly collapse letter + dot + optional space + letter
    return re.sub(r'\b([a-zA-Z])\.(?:\s*([a-zA-Z])\.?)+\b', lambda m: m.group(0).replace('.', '').replace(' ', ''), text)

for s in ["Ferrari N.V.", "Société Générale S.A.", "U.S.A. Inc.", "M.G. Road"]:
    print(f"{s:<25} -> {clean_dotted_acronyms(s)}")
