from dataclasses import asdict, dataclass
import re
import unicodedata

import pandas as pd

MISSING_TEXT = {"", "none", "null", "nan"}
PLUS_CODE_RE = re.compile(r"\b[23456789CFGHJMPQRVWX]{2,}\+[\w\d]+\b", re.IGNORECASE)
DIGIT_RE = re.compile(r"\d")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class AreaNormalization:
    area: str | None
    area_original: str | None
    area_quality_flag: str
    sub_area: str | None


def normalize_location_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = " ".join(text.split())

    if text.lower() in MISSING_TEXT:
        return None

    return text.title()


def location_parts(value: str | None) -> list[str]:
    text = normalize_location_text(value)
    if text is None:
        return []

    return [
        part_normalized
        for part in text.split(",")
        if (part_normalized := normalize_location_text(part)) is not None
    ]


def has_letter(value: str) -> bool:
    return bool(LETTER_RE.search(value))


def is_valid_area_name(value: str | None) -> bool:
    if value is None:
        return False

    text = normalize_location_text(value)
    if text is None:
        return False

    if not has_letter(text):
        return False

    if DIGIT_RE.search(text):
        return False

    if PLUS_CODE_RE.search(text):
        return False

    if len(text) > 60:
        return False

    return True


def best_named_location(value) -> str | None:
    parts = location_parts(value)
    for part in reversed(parts):
        if is_valid_area_name(part):
            return part

    text = normalize_location_text(value)
    if is_valid_area_name(text):
        return text

    return None


def build_area_original(city, area_raw) -> str | None:
    city_text = normalize_location_text(city)
    area_text = normalize_location_text(area_raw)

    if city_text and area_text and city_text.lower() != area_text.lower():
        return f"{city_text} - {area_text}"

    return city_text or area_text


def choose_model_area(city, area_raw) -> AreaNormalization:
    city_area = best_named_location(city)
    raw_area = best_named_location(area_raw)
    raw_text = normalize_location_text(area_raw)
    original = build_area_original(city, area_raw)

    if raw_area and (not city_area or raw_area.lower() != city_area.lower()):
        return AreaNormalization(
            area=raw_area,
            area_original=original,
            area_quality_flag="area_raw_as_area",
            sub_area=None,
        )

    if city_area:
        sub_area = None
        if (
            raw_text
            and raw_text.lower() != city_area.lower()
            and (not raw_area or raw_area.lower() != city_area.lower())
        ):
            sub_area = raw_text

        return AreaNormalization(
            area=city_area,
            area_original=original,
            area_quality_flag="city_as_area_with_sub_area" if sub_area else "city_as_area",
            sub_area=sub_area,
        )

    return AreaNormalization(
        area=None,
        area_original=original,
        area_quality_flag="invalid_area",
        sub_area=raw_text,
    )


def normalize_area_frame(city: pd.Series, area_raw: pd.Series) -> pd.DataFrame:
    records = [
        asdict(choose_model_area(city_value, area_value))
        for city_value, area_value in zip(city, area_raw)
    ]
    return pd.DataFrame(records, index=city.index)
