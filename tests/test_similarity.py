from beneficiary_dedupe.normalize import has_arabic, normalize_arabic, normalize_latin
from beneficiary_dedupe.similarity import name_similarity, sound


def sim(a: str, b: str) -> float:
    na = normalize_arabic(a) if has_arabic(a) else normalize_latin(a)
    nb = normalize_arabic(b) if has_arabic(b) else normalize_latin(b)
    return name_similarity(na, sound(a), has_arabic(a), nb, sound(b), has_arabic(b))


def test_cross_script_same_name_scores_high():
    for latin, arabic in [("Hussein", "حسين"), ("Mohammad", "محمد"), ("Mustafa", "مصطفى"),
                          ("El Khoury", "الخوري"), ("Fatmeh", "فاطمة"), ("Elie", "إيلي")]:
        assert sim(latin, arabic) >= 0.9, (latin, arabic)


def test_long_vowels_separate_names_the_skeleton_merges():
    # Hassan/Hussein share the consonants h-s-n; the Arabic ya marks Hussein.
    assert sim("Hussein", "حسين") - sim("Hassan", "حسين") > 0.1
    assert sim("Hassan", "حسن") - sim("Hussein", "حسن") > 0.1
    assert sim("Mahmoud", "محمود") - sim("Mohammad", "محمود") > 0.1


def test_arabic_one_letter_difference_is_not_a_match():
    assert sim("محمد", "محمود") < 0.85
    assert sim("محمد", "مُحَمَّد") == 1.0


def test_symmetric():
    assert sim("Hussein", "حسين") == sim("حسين", "Hussein")
