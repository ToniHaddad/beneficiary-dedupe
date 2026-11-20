import pytest

from beneficiary_dedupe.normalize import (
    DOB, compare_dob, normalize_arabic, normalize_latin, normalize_phone,
    parse_dob, phonetic, skeleton, soundex,
)

# ---------------------------------------------------------------- Arabic rules


@pytest.mark.parametrize("raw, expected", [
    ("أحمد", "احمد"),          # alef with hamza above
    ("إبراهيم", "ابراهيم"),    # alef with hamza below
    ("آمنة", "امنه"),          # alef madda, and taa marbuta
    ("فاطمة", "فاطمه"),        # taa marbuta -> ha
    ("مصطفى", "مصطفي"),        # alef maqsura -> ya
    ("مُحَمَّد", "محمد"),       # diacritics (damma, fatha, shadda)
    ("محـــمد", "محمد"),       # tatweel
    ("وائل", "وايل"),          # hamza on ya
    ("مؤمن", "مومن"),          # hamza on waw
    ("  علي   حسن ", "علي حسن"),  # spaces
])
def test_arabic_normalisation(raw, expected):
    assert normalize_arabic(raw) == expected


def test_latin_normalisation_drops_accents_articles_and_punctuation():
    assert normalize_latin("Chérif") == "cherif"
    assert normalize_latin("El-Khoury") == "khoury"
    assert normalize_latin("Al Masri") == "masri"
    assert normalize_latin("Wa'el") == "wael"


# ---------------------------------------------------------------- shared key

SAME_NAME = [
    ("Mohammad", "محمد"), ("Mohamed", "محمد"), ("Muhammed", "محمد"),
    ("Hussein", "حسين"), ("Hussain", "حسين"), ("Hsein", "حسين"),
    ("Khoury", "الخوري"), ("El Khoury", "الخوري"), ("Al-Khouri", "الخوري"),
    ("Hajj", "الحاج"), ("Hage", "الحاج"),
    ("Chehab", "شهاب"), ("Shehab", "شهاب"),
    ("Fatmeh", "فاطمة"), ("Fatima", "فاطمة"),
    ("Abdallah", "عبد الله"), ("Abd Allah", "عبدالله"),
    ("Kassem", "قاسم"), ("Qasem", "قاسم"),
    ("Mustafa", "مصطفى"), ("Ziyad", "زياد"), ("Darwiche", "درويش"),
    ("Ibrahim", "إبراهيم"), ("Brahim", "ابراهيم"),
    ("Bar Elyas", "برالياس"), ("Kfar Roummane", "كفررمان"),
]


@pytest.mark.parametrize("latin, arabic", SAME_NAME)
def test_skeleton_brings_arabic_and_latin_together(latin, arabic):
    assert skeleton(latin) == skeleton(arabic)


@pytest.mark.parametrize("a, b", [
    ("Mohammad", "Khaled"), ("Hassan", "Ali"), ("Khoury", "Haddad"), ("محمد", "خالد"),
])
def test_skeleton_keeps_different_names_apart(a, b):
    assert skeleton(a) != skeleton(b)


def test_phonetic_key_beats_soundex_where_soundex_fails():
    # Soundex keeps the first letter as written and does not know the article.
    for a, b in [("Chehab", "Shehab"), ("Kassem", "Qasem"), ("Khoury", "El Khoury"), ("Hajj", "Hage")]:
        assert phonetic(a) == phonetic(b)
    assert soundex("Chehab") != soundex("Shehab")
    assert soundex("Khoury") != soundex("El Khoury")
    assert soundex("Robert") == "R163"  # the reference implementation is standard


def test_georges_is_a_known_miss():
    # French silent final s: documented in "Where it fails".
    assert skeleton("Georges") != skeleton("جورج")
    assert skeleton("George") == skeleton("جورج")


# ---------------------------------------------------------------- dates


def test_parse_dob_formats():
    assert parse_dob("1985-03-07") == DOB(1985, 3, 7)
    assert parse_dob("07/03/1985") == DOB(1985, 3, 7)     # day first
    assert parse_dob("7/3/1985") == DOB(1985, 3, 7)
    assert parse_dob("١٩٨٥-٠٣-٠٧") == DOB(1985, 3, 7)   # Arabic-Indic digits
    assert parse_dob("1985") == DOB(1985)
    assert parse_dob("12/25/1985") == DOB(1985, 12, 25)   # month first, unambiguous
    assert parse_dob("") == DOB(None)
    assert parse_dob("unknown") == DOB(None)


@pytest.mark.parametrize("a, b, level", [
    ("07/03/1985", "1985-03-07", "exact"),
    ("07/03/1985", "03/07/1985", "swapped"),
    ("1985-03-07", "1985-07-03", "swapped"),
    ("05/05/1985", "05/05/1985", "exact"),        # day == month: no swap to detect
    ("1985", "07/03/1985", "year_only"),
    ("01/01/1985", "07/03/1985", "same_year"),
    ("07/03/1986", "07/03/1985", "near_year"),
    ("07/03/1990", "07/03/1985", "differs"),
    ("", "07/03/1985", "missing"),
])
def test_compare_dob(a, b, level):
    assert compare_dob(parse_dob(a), parse_dob(b)) == level
    assert compare_dob(parse_dob(b), parse_dob(a)) == level


# ---------------------------------------------------------------- phones


@pytest.mark.parametrize("raw", [
    "+961 3 123 456", "03123456", "03-123456", "961 3123456", "00961 3 123456",
    "3123456", "+9613123456", "٠٣١٢٣٤٥٦", "03/123 456",
])
def test_phone_formats_normalise_to_one_number(raw):
    assert normalize_phone(raw) == "3123456"


def test_eight_digit_mobile_and_rubbish():
    assert normalize_phone("+961 71 234 567") == "71234567"
    assert normalize_phone("71-234567") == "71234567"
    assert normalize_phone("") == ""
    assert normalize_phone("123") == ""
