"""Name similarity within and across scripts.

The consonant skeleton is good for blocking but too coarse to score with:
Hassan and Hussein (حسن / حسين) share "hsn", Mohammad and Mahmoud share
"mhmd". The difference is in the long vowels. So each name is also turned
into a "sound" string: consonant classes plus vowel classes (a, i, u).

* Latin: single vowels are short (lower case); digraphs such as "ou", "ee",
  "ei", "ai" are long (upper case).
* Arabic: only long vowels are written (ا, ي, و), always upper case.

Across scripts the two sound strings are aligned with a weighted edit
distance where a Latin short vowel costs almost nothing to insert (Arabic
does not write it), but an Arabic long vowel with nothing to match it, or a
Latin long vowel with no Arabic letter under it, costs a lot.
"""

from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz.distance import JaroWinkler

from .normalize import (
    ARABIC_TO_KEY, ARTICLE_EXCEPTIONS, LATIN_SINGLE, has_arabic,
    normalize_arabic, normalize_latin,
)

LATIN_SOUND_DIGRAPHS = [
    ("kh", "X"), ("sh", "S"), ("ch", "S"), ("gh", "G"), ("th", "t"),
    ("dh", "d"), ("ph", "f"), ("ck", "k"), ("dj", "j"),
    ("ou", "W"), ("oo", "W"), ("ee", "I"), ("ei", "I"), ("ie", "I"),
    ("ai", "I"), ("ay", "I"), ("ey", "I"), ("aa", "A"),
]
SHORT = {"a": "a", "e": "e", "i": "i", "o": "u", "u": "u"}
LONG = set("AIUW")
SHORT_SET = set("aeiu")
NEAR = {frozenset(p) for p in [("s", "S"), ("s", "z"), ("k", "X"), ("k", "G"), ("t", "d"), ("X", "h"), ("j", "S")]}


def _latin_token_sound(token: str) -> str:
    for src, dst in LATIN_SOUND_DIGRAPHS:
        token = token.replace(src, dst)
    out = []
    for i, ch in enumerate(token):
        if ch in SHORT:
            out.append(SHORT[ch])
        elif ch == "y":
            out.append("y" if i == 0 else "i")
        elif ch == "w":
            out.append("w" if i == 0 else "u")
        else:
            out.append(LATIN_SINGLE.get(ch, ch))
    return "".join(out)


def _arabic_token_sound(token: str) -> str:
    if token.startswith("ال") and len(token) >= 4 and token not in ARTICLE_EXCEPTIONS:
        token = token[2:]
    out = []
    for i, ch in enumerate(token):
        if ch == "و":
            out.append("w" if i == 0 else "U")
        elif ch == "ي":
            out.append("y" if i == 0 else "I")
        elif ch == "ا":
            out.append("" if i == 0 else "A")
        else:
            out.append(ARABIC_TO_KEY.get(ch, ""))
    return "".join(out)


def sound(name: str) -> str:
    """Consonant + vowel-class string for a name in either script."""
    if not name:
        return ""
    if has_arabic(name):
        # Alef maqsura is read "a" (مصطفى Mustafa, عيسى Issa): keep it as alef
        # before the general normalisation turns it into ya.
        parts = [_arabic_token_sound(t) for t in normalize_arabic(name.replace("ى", "ا")).split()]
    else:
        parts = [_latin_token_sound(t) for t in normalize_latin(name).split()]
    return re.sub(r"(.)\1+", r"\1", "".join(parts))


VOWEL_CLASS = {"a": "a", "A": "a", "i": "i", "I": "i", "u": "u", "U": "u", "W": "u", "e": "e"}


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ca, cb = VOWEL_CLASS.get(a), VOWEL_CLASS.get(b)
    if ca and cb:
        if ca == cb or ("e" in (ca, cb) and {ca, cb} <= {"e", "a", "i"}):
            return 0.0  # Latin "e" stands for either fatha or kasra
        return 0.6
    if ca or cb:
        return 2.0
    return 0.5 if frozenset((a, b)) in NEAR else 1.0


def _del_arabic(a: str, last: bool) -> float:
    if a == "h" and last:
        return 0.15  # taa marbuta or final ha left out in Latin ("Fatima")
    return 0.6 if a in LONG else 1.0


def _ins_latin(b: str, last: bool) -> float:
    if b in SHORT_SET:
        return 0.05
    if b == "W":
        return 0.3  # "ou" writes both short and long u in French-style spelling
    if b in LONG:
        return 0.4
    if b == "h":
        return 0.15 if last else 0.7
    return 1.0


@lru_cache(maxsize=200_000)
def cross_script_similarity(arabic_sound: str, latin_sound: str) -> float:
    """Similarity in [0, 1] between an Arabic and a Latin sound string."""
    x, y = arabic_sound, latin_sound
    if not x or not y:
        return 0.0
    n, m = len(x), len(y)
    ins = [_ins_latin(y[j], j == m - 1) for j in range(m)]
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + ins[j - 1]
    for i in range(1, n + 1):
        d = _del_arabic(x[i - 1], i == n)
        cur = [prev[0] + d] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j - 1] + _sub_cost(x[i - 1], y[j - 1]),
                prev[j] + d,
                cur[j - 1] + ins[j - 1],
            )
        prev = cur
    weight = len(x) + sum(1 for c in y if c not in VOWEL_CLASS)
    return max(0.0, 1.0 - 2.0 * prev[m] / weight)

@lru_cache(maxsize=200_000)
def _same_script(a_sound: str, b_sound: str, a_text: str, b_text: str) -> float:
    by_sound = JaroWinkler.normalized_similarity(a_sound.lower(), b_sound.lower())
    by_text = JaroWinkler.normalized_similarity(a_text, b_text)
    return max(by_sound, by_text)


def name_similarity(a_text: str, a_sound: str, a_arabic: bool,
                    b_text: str, b_sound: str, b_arabic: bool) -> float:
    """Similarity of two name parts, whatever script each is written in.

    ``*_text`` is the same-script normalised text, ``*_sound`` the output of
    :func:`sound`.
    """
    if not a_sound or not b_sound:
        return 0.0
    if a_arabic == b_arabic:
        return _same_script(a_sound, b_sound, a_text, b_text)
    if a_arabic:
        return cross_script_similarity(a_sound, b_sound)
    return cross_script_similarity(b_sound, a_sound)
