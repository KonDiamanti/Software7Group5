# cleaning.py
# text cleaning and the tamper rule for tier 1

import unicodedata

# zero-width (invisible) characters, we strip these out
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# cyrillic lookalike letters -> normal latin letters (written as codes so copy and paste cannot break them)
CYRILLIC = "\u0430\u0435\u043e\u0440\u0441\u0445\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0425"
LATIN = "aeopcxABEKMHOPCTX"
HOMOGLYPHS = str.maketrans(CYRILLIC, LATIN)


def clean_lower(text):
    text = unicodedata.normalize("NFKC", text)   # also fixes full-width letters
    text = text.translate(ZERO_WIDTH).translate(HOMOGLYPHS)
    return text.lower()


def suspicious(text):
    # tamper rule: invisible characters, or a word that mixes alphabets
    if any(unicodedata.category(ch) == "Cf" for ch in text):
        return True
    for word in text.split():
        scripts = {unicodedata.name(ch, "UNKNOWN").split()[0] for ch in word if ch.isalpha()}
        if len(scripts) > 1:   # e.g. LATIN + CYRILLIC
            return True
    return False