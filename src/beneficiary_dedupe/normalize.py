"""Normalisation for names, dates of birth and phone numbers.

Three kinds of key are built from a name:

* ``normalize_arabic`` / ``normalize_latin``: the same script, cleaned up
  (hamza forms, taa marbuta, diacritics, accents, articles).
* ``skeleton``: a consonant skeleton that works on BOTH scripts. Arabic
  letters and Latin spellings are mapped to the same consonant classes and
  vowels are dropped, so "Mohamed El Khouri" and "محمد الخوري" give the
  same skeleton ("mhmdXr").
* ``phonetic``: a coarser code built on the skeleton for blocking. It merges
  letters that Latin transliteration confuses (kh/k/q, sh/ch/s/z) so a
  variant spelling still lands in the same block. English Soundex does badly
  here because it keeps the first letter as written ("Chehab" C110 against
  "Shehab" S110) and does not know about the article ("El Khoury").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------- Arabic

ARABIC_RE = re.compile(r"[؀-ۿ]")
DIACRITICS_RE = re.compile(r"[ً-ٰٟۖ-ۭ]")
TATWEEL = "ـ"

ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
OTHER_FORMS = str.maketrans({"ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي"})
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Words that start with the letters of the article but where the article is
# part of the name itself.
ARTICLE_EXCEPTIONS = {"الله", "الياس", "اليسار"}

# Arabic letter -> consonant class. Long vowels and ayn/hamza map to "".
ARABIC_TO_KEY = {
    "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "h", "خ": "X", "د": "d",
    "ذ": "d", "ر": "r", "ز": "z", "س": "s", "ش": "S", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z", "ع": "", "غ": "G", "ف": "f", "ق": "k", "ك": "k",
    "ل": "l", "م": "m", "ن": "n", "ه": "h", "ا": "", "ء": "",
    "پ": "b", "چ": "S", "ڤ": "f", "گ": "j",
}

# Latin articles written as their own token or with a hyphen.
LATIN_ARTICLES = {"el", "al", "ul", "il"}
LATIN_DIGRAPHS = [
    ("kh", "X"), ("sh", "S"), ("ch", "S"), ("gh", "G"), ("th", "t"),
    ("dh", "d"), ("ph", "f"), ("ck", "k"), ("dj", "j"), ("ou", "u"),
]
LATIN_SINGLE = {"q": "k", "c": "k", "g": "j", "v": "f", "p": "b", "x": "ks"}
VOWELS = set("aeiou")


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def normalize_arabic(text: str) -> str:
    """Remove tatweel and diacritics, unify alef/hamza, taa marbuta, alef maqsura."""
    text = (text or "").replace(TATWEEL, "")
    text = DIACRITICS_RE.sub("", text)
    text = text.translate(ALEF_FORMS).translate(OTHER_FORMS)
    text = text.replace("ء", "")
    text = re.sub(r"[^؀-ۿ0-9 ]+", " ", text)
    return " ".join(text.split())


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_latin(text: str) -> str:
    """Lower case, no accents, no punctuation, no article tokens."""
    text = strip_accents(text or "").lower()
    text = re.sub(r"[-_.]", " ", text)
    text = re.sub(r"[^a-z ]+", "", text)
    tokens = [t for t in text.split() if t not in LATIN_ARTICLES]
    return " ".join(tokens)


def _arabic_token_key(token: str) -> str:
    if token.startswith("ال") and len(token) >= 4 and token not in ARTICLE_EXCEPTIONS:
        token = token[2:]
    out = []
    for i, ch in enumerate(token):
        if ch in ("و", "ي"):
            out.append(("w" if ch == "و" else "y") if i == 0 else "")
        else:
            out.append(ARABIC_TO_KEY.get(ch, ""))
    return "".join(out)


def _latin_token_key(token: str) -> str:
    for src, dst in LATIN_DIGRAPHS:
        token = token.replace(src, dst)
    out = []
    for i, ch in enumerate(token):
        if ch in VOWELS:
            continue
        if ch in ("y", "w"):
            if i == 0:
                out.append(ch)
            continue
        out.append(LATIN_SINGLE.get(ch, ch))
    return "".join(out)


def _finish(key: str) -> str:
    # Word-final h is dropped in both scripts: taa marbuta (فاطمة / Fatmeh /
    # Fatima) and a final ha/haa are written inconsistently in Latin.
    key = key[:-1] if key.endswith("h") else key
    return re.sub(r"(.)\1+", r"\1", key)


def skeleton(name: str) -> str:
    """Consonant skeleton shared by Arabic and Latin spellings of a name.

    Multi-word name parts ("Abd Allah", "El Khoury") are joined, so the
    result does not depend on where a writer put the spaces.
    """
    if not name:
        return ""
    if has_arabic(name):
        tokens = normalize_arabic(name).split()
        keys = [_finish(_arabic_token_key(t)) for t in tokens]
    else:
        tokens = normalize_latin(name).split()
        keys = [_finish(_latin_token_key(t)) for t in tokens]
    return re.sub(r"(.)\1+", r"\1", "".join(keys))


PHONETIC_MERGE = str.maketrans({"X": "k", "G": "k", "S": "s", "z": "s", "t": "d"})


def phonetic(name: str) -> str:
    """Coarse phonetic code for blocking, built on the skeleton."""
    key = skeleton(name).translate(PHONETIC_MERGE)
    return re.sub(r"(.)\1+", r"\1", key)


def soundex(name: str) -> str:
    """Plain American Soundex, kept only to compare against in the benchmark."""
    name = re.sub(r"[^a-z]", "", strip_accents(name or "").lower())
    if not name:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    out, last = name[0].upper(), codes.get(name[0], "")
    for ch in name[1:]:
        code = codes.get(ch, "")
        if code and code != last:
            out += code
        if ch not in "hw":
            last = code
    return (out + "000")[:4]


# ---------------------------------------------------------------- dates

@dataclass(frozen=True)
class DOB:
    year: int | None
    month: int | None = None
    day: int | None = None

    @property
    def year_only(self) -> bool:
        return self.year is not None and self.month is None

    def iso(self) -> str:
        if self.year is None:
            return ""
        if self.month is None:
            return f"{self.year:04d}"
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"


def parse_dob(text: str) -> DOB:
    """Parse '1985-03-07', '07/03/1985' (day first), '7-3-1985' or '1985'."""
    text = (text or "").translate(ARABIC_DIGITS).strip()
    if not text:
        return DOB(None)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        return DOB(y, mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else DOB(y)
    m = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", text)
    if m:
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
        if mo > 12 and d <= 12:  # written month first by mistake
            d, mo = mo, d
        return DOB(y, mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else DOB(y)
    m = re.fullmatch(r"(\d{4})", text)
    if m:
        return DOB(int(m[1]))
    return DOB(None)


DOB_LEVELS = ("exact", "swapped", "year_only", "same_year", "near_year", "differs", "missing")


def compare_dob(a: DOB, b: DOB) -> str:
    """Classify how two dates of birth agree.

    "swapped" means the same date with day and month exchanged
    (07/03/1985 against 03/07/1985), which happens when one team writes
    day first and another month first.
    """
    if a.year is None or b.year is None:
        return "missing"
    if a.year == b.year:
        if a.year_only or b.year_only:
            return "year_only"
        if (a.month, a.day) == (b.month, b.day):
            return "exact"
        if a.day == b.month and a.month == b.day:
            return "swapped"
        return "same_year"
    if abs(a.year - b.year) == 1:
        return "near_year"
    return "differs"


# ---------------------------------------------------------------- phones

def normalize_phone(text: str) -> str:
    """Return the Lebanese national number without trunk zero, or ''.

    '+961 3 123 456', '00961-3-123456', '03123456', '03/123 456' and
    '٠٣١٢٣٤٥٦' all give '3123456'. '+961 71 234 567' and '71234567' give
    '71234567'.
    """
    digits = re.sub(r"\D", "", (text or "").translate(ARABIC_DIGITS))
    if digits.startswith("00961"):
        digits = digits[5:]
    elif digits.startswith("961") and len(digits) >= 10:
        digits = digits[3:]
    digits = digits.lstrip("0")
    if len(digits) < 7 or len(digits) > 8:
        return ""
    return digits
