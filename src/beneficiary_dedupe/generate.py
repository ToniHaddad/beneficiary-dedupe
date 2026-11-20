"""Seeded generator of synthetic household registrations for a fictional response.

Every record is invented. Duplicates are made the way they arise in the field:
a second team registers the same household with another transliteration, in
Arabic script, with the names in another order, without the father's name,
with day and month swapped, with a year only or an estimated age, with the
phone written another way or changed, in another village after moving, with
the household size off by one or after a split, or with a typing error. Hard
negatives are added on purpose: siblings (and twins) who head their own
households, and different people with the same name in the same village.

Ground truth is in ``cluster_id`` (one id per real household), ``gen_note``
(what was done to the record) and ``related_to`` (for a sibling or namesake,
the household it was made to resemble). The matcher never reads them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from . import names
from .records import FIELDS, TRUTH_FIELDS

ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
MOBILE_PREFIXES = ["3", "3", "3", "70", "71", "76", "78", "79", "81"]


@dataclass(frozen=True)
class Person:
    cluster_id: str
    female: bool
    given: int
    father: int
    family: int
    year: int
    month: int
    day: int
    phone: str
    village: int
    household: int


def _weighted(rng: random.Random, table) -> int:
    return rng.choices(range(len(table)), weights=[t[2] for t in table])[0]


def _latin_variant(rng: random.Random, variants: list[str]) -> str:
    if len(variants) == 1 or rng.random() < 0.45:
        return variants[0]
    return rng.choice(variants[1:])


def _new_phone(rng: random.Random) -> str:
    prefix = rng.choice(MOBILE_PREFIXES)
    return prefix + "".join(str(rng.randint(0, 9)) for _ in range(6))


def _render_phone(rng: random.Random, number: str, arabic: bool) -> str:
    pfx = number[:-6]
    rest = number[-6:]
    national = ("0" + number) if len(number) == 7 else number
    styles = [
        f"+961 {pfx} {rest[:3]} {rest[3:]}",
        national,
        f"{national[:-6]}-{rest}",
        f"961{number}",
        f"00961 {number}",
        f"{national[:-6]} {rest[:3]} {rest[3:]}",
    ]
    text = rng.choice(styles)
    if arabic and rng.random() < 0.4:
        text = national.translate(ARABIC_DIGITS)
    return text


def _render_dob(rng: random.Random, p: Person, kind: str, arabic: bool) -> str:
    y, m, d = p.year, p.month, p.day
    if kind == "year_only":
        text = str(y)
    elif kind == "jan1":
        text = rng.choice([f"01/01/{y}", f"{y}-01-01"])
    else:
        if kind == "swapped":
            m, d = d, m
        text = rng.choice([f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}", f"{d}/{m}/{y}"])
    if arabic and rng.random() < 0.3:
        text = text.translate(ARABIC_DIGITS)
    return text


def _arabic_variation(rng: random.Random, text: str) -> str:
    """Orthographic variation real Arabic data entry shows."""
    if rng.random() < 0.5:
        text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    if rng.random() < 0.4:
        text = text.replace("ة", "ه")
    if rng.random() < 0.4:
        text = text.replace("ى", "ي")
    if rng.random() < 0.15 and len(text) > 3:
        text = text[:2] + "ـ" + text[2:]  # tatweel
    if rng.random() < 0.15:
        text = text.replace("محمد", "مُحَمَّد")
    return text


def _typo(rng: random.Random, text: str, arabic: bool = False) -> str:
    letters = [i for i, c in enumerate(text) if c.isalpha()]
    if len(letters) < 3:
        return text
    i = rng.choice(letters[1:])
    op = rng.choice(["delete", "insert", "substitute", "transpose"])
    insert_pool, sub_pool = ("اويهنرس", "اويمنرسل") if arabic else ("aeiouhnrs", "aeioumnrsl")
    if op == "delete":
        return text[:i] + text[i + 1:]
    if op == "insert":
        return text[:i] + rng.choice(insert_pool) + text[i:]
    if op == "substitute":
        return text[:i] + rng.choice(sub_pool) + text[i + 1:]
    if i + 1 < len(text) and text[i + 1].isalpha():
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text


class Generator:
    def __init__(self, seed: int = 1):
        self.rng = random.Random(seed)
        self.next_cluster = 0

    # ------------------------------------------------------------ persons
    def _cluster_id(self) -> str:
        self.next_cluster += 1
        return f"H{self.next_cluster:06d}"

    def new_person(self) -> Person:
        rng = self.rng
        female = rng.random() < 0.25
        table = names.FEMALE_GIVEN if female else names.MALE_GIVEN
        return Person(
            cluster_id=self._cluster_id(),
            female=female,
            given=_weighted(rng, table),
            father=_weighted(rng, names.MALE_GIVEN),
            family=_weighted(rng, names.FAMILY),
            year=rng.randint(1950, 2004),
            month=rng.randint(1, 12),
            day=rng.randint(1, 28),
            phone=_new_phone(rng),
            village=rng.randrange(len(names.VILLAGES)),
            household=max(1, min(12, int(rng.gauss(4.8, 2.0)))),
        )

    def sibling(self, p: Person, twin: bool) -> Person:
        rng = self.rng
        female = rng.random() < 0.25
        table = names.FEMALE_GIVEN if female else names.MALE_GIVEN
        given = _weighted(rng, table)
        while given == p.given and female == p.female:
            given = _weighted(rng, table)
        year = p.year if twin else p.year + rng.choice([-1, 1]) * rng.randint(2, 12)
        return replace(
            p, cluster_id=self._cluster_id(), female=female, given=given, year=year,
            month=p.month if twin else rng.randint(1, 12),
            day=p.day if twin else rng.randint(1, 28),
            phone=p.phone if rng.random() < 0.3 else _new_phone(rng),
            household=max(1, min(12, int(rng.gauss(4.8, 2.0)))),
        )

    def namesake(self, p: Person, same_father: bool) -> Person:
        rng = self.rng
        father = p.father if same_father else _weighted(rng, names.MALE_GIVEN)
        return replace(
            p, cluster_id=self._cluster_id(), father=father,
            year=p.year + rng.choice([-1, 1]) * rng.randint(1, 20),
            month=rng.randint(1, 12), day=rng.randint(1, 28),
            phone=_new_phone(rng),
            household=max(1, min(12, int(rng.gauss(4.8, 2.0)))),
        )

    # ------------------------------------------------------------ records
    def render(self, p: Person, arabic: bool, tags: list[str], note: str, related: str = "") -> dict:
        rng = self.rng
        gtable = names.FEMALE_GIVEN if p.female else names.MALE_GIVEN
        village = p.village
        if "moved" in tags:  # displaced again and registered where they are now
            village = (village + rng.randrange(1, len(names.VILLAGES))) % len(names.VILLAGES)
        district, village_ar, village_lat = names.VILLAGES[village]

        def name(table, i):
            ar, lat, _ = table[i]
            return _arabic_variation(rng, ar) if arabic else _latin_variant(rng, lat)

        given, father, family = name(gtable, p.given), name(names.MALE_GIVEN, p.father), name(names.FAMILY, p.family)
        if "typo" in tags:
            which = rng.choice(["given", "family", "father"])
            if which == "given":
                given = _typo(rng, given, arabic)
            elif which == "family":
                family = _typo(rng, family, arabic)
            else:
                father = _typo(rng, father, arabic)
        if "no_father" in tags:
            father = ""
        if "swap_order" in tags:
            given, family = family, given

        dob_kind = next((t for t in ("swapped", "year_only", "jan1") if t in tags), "plain")
        person = p
        if "year_typo" in tags:
            person = replace(p, year=p.year + rng.choice([-1, 1]))
        if "age_estimated" in tags:  # no papers: age guessed by the registrar
            person = replace(p, year=p.year + rng.choice([-1, 1]) * rng.randint(2, 5))
            dob_kind = rng.choice(["year_only", "jan1"])
        dob = _render_dob(rng, person, dob_kind, arabic)

        phone = "" if "no_phone" in tags else _render_phone(rng, p.phone, arabic)
        if "new_phone" in tags:
            phone = _render_phone(rng, _new_phone(rng), arabic)
        household = p.household
        if "household_pm1" in tags:
            household = max(1, household + rng.choice([-1, 1]))
        if "household_split" in tags:
            household = max(1, household + rng.choice([-3, -2, 2, 3]))

        team_pool = sorted(names.ARABIC_TEAMS) if arabic else [t for t in names.TEAMS if t not in names.ARABIC_TEAMS]
        return {
            "given_name": given,
            "father_name": father,
            "family_name": family,
            "date_of_birth": dob,
            "phone": phone,
            "village": _arabic_variation(rng, village_ar) if arabic else _latin_variant(rng, village_lat),
            "district": district,
            "household_size": str(household),
            "team": rng.choice(team_pool),
            "cluster_id": p.cluster_id,
            "gen_note": note,
            "related_to": related,
        }

    def household_records(self, p: Person, kind: str, related: str = "") -> list[dict]:
        rng = self.rng
        base_arabic = rng.random() < 0.25
        base_tags = []
        if rng.random() < 0.12:
            base_tags.append("no_father")
        if rng.random() < 0.05:
            base_tags.append("year_only")
        if rng.random() < 0.04:
            base_tags.append("no_phone")
        out = [self.render(p, base_arabic, base_tags, kind, related)]

        if rng.random() < 0.30:
            n_dups = rng.choices([1, 2, 3], weights=[70, 22, 8])[0]
            for _ in range(n_dups):
                # A Latin record gets an Arabic copy 30% of the time; an
                # Arabic record is transliterated 60% of the time.
                arabic = (rng.random() >= 0.6) if base_arabic else (rng.random() < 0.30)
                tags = []
                if arabic != base_arabic:
                    tags.append("arabic" if arabic else "latin")
                if rng.random() < 0.10:
                    tags.append("swap_order")
                if "no_father" in base_tags:
                    if rng.random() < 0.6:
                        tags.append("added_father")
                    else:
                        tags.append("no_father")
                elif rng.random() < 0.15:
                    tags.append("no_father")
                r = rng.random()
                if r < 0.12 and p.day <= 12 and p.day != p.month:
                    tags.append("swapped")
                elif r < 0.20:
                    tags.append("year_only")
                elif r < 0.24:
                    tags.append("jan1")
                elif r < 0.27:
                    tags.append("year_typo")
                elif r < 0.31:
                    tags.append("age_estimated")
                if rng.random() < 0.08:
                    tags.append("moved")
                if rng.random() < 0.10:
                    tags.append("new_phone")
                elif rng.random() < 0.07:
                    tags.append("no_phone")
                h = rng.random()
                if h < 0.20:
                    tags.append("household_pm1")
                elif h < 0.25:
                    tags.append("household_split")
                if rng.random() < (0.08 if arabic else 0.18):
                    tags.append("typo")
                note = "dup:" + ",".join(tags) if tags else "dup:respelled"
                out.append(self.render(p, arabic, tags, note, related))
        return out

    def records(self, n: int) -> list[dict]:
        rng = self.rng
        rows: list[dict] = []
        while len(rows) < n:
            p = self.new_person()
            rows.extend(self.household_records(p, "base"))
            if rng.random() < 0.06:
                twin = rng.random() < 0.15
                rows.extend(self.household_records(self.sibling(p, twin), "neg:twin" if twin else "neg:sibling", p.cluster_id))
            if rng.random() < 0.05:
                same_father = rng.random() < 0.25
                kind = "neg:namesake_same_father" if same_father else "neg:namesake"
                rows.extend(self.household_records(self.namesake(p, same_father), kind, p.cluster_id))
        rows = rows[:n]
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            row["record_id"] = f"NBR-{i + 1:06d}"
        return [{k: row[k] for k in FIELDS + TRUTH_FIELDS} for row in rows]


def generate(n: int = 10_000, seed: int = 1) -> list[dict]:
    """Return ``n`` synthetic registration rows, reproducible from ``seed``."""
    return Generator(seed).records(n)
