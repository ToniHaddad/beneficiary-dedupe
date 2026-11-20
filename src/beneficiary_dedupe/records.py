"""Record loading, saving and the prepared (normalised) view used for matching."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .normalize import (
    DOB, has_arabic, normalize_arabic, normalize_latin, normalize_phone,
    parse_dob, phonetic, skeleton,
)
from .similarity import sound

FIELDS = [
    "record_id", "given_name", "father_name", "family_name", "date_of_birth",
    "phone", "village", "district", "household_size", "team",
]
TRUTH_FIELDS = ["cluster_id", "gen_note", "related_to"]


@dataclass
class Prepared:
    """One record with every key the matcher needs, computed once."""

    idx: int
    raw: dict
    arabic: bool
    given: str          # same-script normalised text
    father: str
    family: str
    given_sk: str       # cross-script skeletons
    father_sk: str
    family_sk: str
    given_snd: str      # consonant + vowel-class strings for scoring
    father_snd: str
    family_snd: str
    given_ph: str       # phonetic codes for blocking
    family_ph: str
    dob: DOB
    phone: str
    village_sk: str
    district: str
    household: int | None


def _norm(text: str) -> str:
    return normalize_arabic(text) if has_arabic(text) else normalize_latin(text)


def prepare(idx: int, row: dict) -> Prepared:
    given, father, family = row.get("given_name", ""), row.get("father_name", ""), row.get("family_name", "")
    try:
        household = int(str(row.get("household_size", "")).strip())
    except ValueError:
        household = None
    given_sk, father_sk, family_sk = skeleton(given), skeleton(father), skeleton(family)
    return Prepared(
        idx=idx,
        raw=row,
        arabic=has_arabic(given + family),
        given=_norm(given),
        father=_norm(father),
        family=_norm(family),
        given_sk=given_sk,
        father_sk=father_sk,
        family_sk=family_sk,
        given_snd=sound(given),
        father_snd=sound(father),
        family_snd=sound(family),
        given_ph=phonetic(given),
        family_ph=phonetic(family),
        dob=parse_dob(row.get("date_of_birth", "")),
        phone=normalize_phone(row.get("phone", "")),
        village_sk=skeleton(row.get("village", "")),
        district=(row.get("district") or "").strip().lower(),
        household=household,
    )


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0].keys()) if rows else FIELDS)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
