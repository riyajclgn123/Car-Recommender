"""
api/routes/parse.py
Free-text intake with NO LLM and NO API key.
Rule-based extraction → validated profile for /recommend.
Anything it had to guess is listed in "assumed" so the user can fix it.
"""
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError
from api.schemas import RecommendRequest

router = APIRouter()


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=600)


STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
STATE_CODES = set(STATE_NAMES.values())
# Codes that are also common English words — only trust them after a comma
AMBIGUOUS_CODES = {"OK", "IN", "ME", "OR", "HI"}

CITY_STATE = {
    "dallas": "TX", "houston": "TX", "austin": "TX", "san antonio": "TX",
    "lubbock": "TX", "new orleans": "LA", "baton rouge": "LA",
    "hammond": "LA", "los angeles": "CA", "san francisco": "CA",
    "san diego": "CA", "new york": "NY", "chicago": "IL", "miami": "FL",
    "orlando": "FL", "tampa": "FL", "atlanta": "GA", "phoenix": "AZ",
    "seattle": "WA", "denver": "CO", "boston": "MA", "nashville": "TN",
    "detroit": "MI",
}

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8}
NUM = r"(\d+|" + "|".join(NUM_WORDS) + r")"

MONEY = re.compile(
    r"(\$)?(\d[\d,]*(?:\.\d+)?)\s*(k\b|grand\b)?"
    r"(?:\s*(?:/|a|an|per|each)?\s*(year|yr|annual(?:ly)?|month|monthly|mo)\b)?"
)
INCOME_WORDS = re.compile(r"\b(make|making|earn|earning|income|salary|paid)\b")


def _num(s):
    return int(s) if s.isdigit() else NUM_WORDS[s]


def _to_monthly(m):
    _, num, k, period = m.groups()
    v = float(num.replace(",", ""))
    if k:
        v *= 1000
    if period:
        return v if period.startswith("mo") else v / 12
    return v / 12 if v >= 15_000 else v   # big bare numbers are annual


def find_monthly_income(t):
    matches = list(MONEY.finditer(t))
    kw_ends = [k.end() for k in INCOME_WORDS.finditer(t)]
    # 1) a number shortly after "make/earn/salary"
    for m in matches:
        if any(0 <= m.start() - e <= 30 for e in kw_ends):
            v = _to_monthly(m)
            if v >= 500:
                return v
    # 2) any number with an explicit "a year" / "a month"
    for m in matches:
        if m.group(4):
            v = _to_monthly(m)
            if v >= 500:
                return v
    return None


def find_age(t):
    for pat in (r"\b(?:i am|i'm|im|age|aged)\s+(\d{2})\b(?!\s*(?:k|,000))",
                r"\b(\d{2})\s*(?:years?\s*old|yrs?\s*old|yo|y/o)\b"):
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    return None


def age_to_group(age):
    if age < 35: return "18-34"
    if age < 55: return "35-54"
    if age < 65: return "55-64"
    return "65+"


def find_household(t):
    m = re.search(rf"family of {NUM}\b", t)
    if m:
        return _num(m.group(1))
    m = re.search(rf"\b{NUM}\s+(?:kids|children|sons|daughters)\b", t)
    if m:
        return 2 + _num(m.group(1))          # assumes two adults
    if re.search(r"\b(married|wife|husband|partner|spouse)\b", t):
        return 2
    return None


def find_state(t, original):
    for city, st in CITY_STATE.items():
        if re.search(rf"\b{city}\b", t):
            return st, True                   # a city → also urban
    for name in sorted(STATE_NAMES, key=len, reverse=True):  # "west virginia" before "virginia"
        if re.search(rf"\b{name}\b", t):
            return STATE_NAMES[name], False
    for m in re.finditer(r"(,\s*)?\b([A-Z]{2})\b", original):
        code = m.group(2)
        if code in STATE_CODES and (code not in AMBIGUOUS_CODES or m.group(1)):
            return code, False
    return None, False


def find_area(t, city_hit):
    if re.search(r"\b(rural|farm|ranch|countryside|small town)\b", t):
        return "rural"
    if re.search(r"\b(suburb|suburbs|suburban)\b", t):
        return "suburban"
    if city_hit or re.search(r"\b(downtown|city|urban|apartment)\b", t):
        return "urban"
    return None


def find_prefer_new(t):
    if re.search(r"\bused\b.{0,12}\b(fine|ok|okay)\b|\bnew or used\b|\beither\b", t):
        return None
    if re.search(r"\b(brand new|new only|only new|a new car)\b", t):
        return True
    if re.search(r"\b(used|pre-owned|preowned|second[- ]hand)\b", t):
        return False
    return None


def find_maintenance(t):
    m = re.search(r"(?:maintenance|repairs?)\D{0,25}(\d[\d,]*)", t)
    if m:
        return float(m.group(1).replace(",", ""))
    if re.search(r"cheap to (fix|maintain|repair)|low maintenance", t):
        return 100.0
    return None


@router.post("/parse-profile")
def parse_profile(req: ParseRequest):
    t = req.text.lower()
    assumed = []

    income = find_monthly_income(t)
    if income is None:
        raise HTTPException(422, "I couldn't find your income. Add something "
                                 "like '$55k a year' or '$4,000 a month'.")

    age = find_age(t)
    if age is not None:
        age_group = age_to_group(age)
    elif re.search(r"\bretire", t):
        age_group = "65+"
    elif re.search(r"\b(student|college)\b", t):
        age_group = "18-34"
    else:
        age_group = "35-54"
        assumed.append("age group 35-54")

    hh = find_household(t)
    if hh is None:
        hh = 1
        assumed.append("household of 1")

    state, city_hit = find_state(t, req.text)
    if state is None:
        state = "TX"
        assumed.append("state TX")

    area = find_area(t, city_hit)
    if area is None:
        area = "suburban"
        assumed.append("suburban area")

    annual = income * 12
    band = "low" if annual < 40_000 else "middle" if annual <= 100_000 else "high"

    data = {
        "monthly_income": round(income, 2),
        "max_maintenance_monthly": find_maintenance(t) or 150,
        "household_size": max(1, min(hh, 10)),
        "age_group": age_group,
        "income_band": band,
        "area_type": area,
        "state": state,
        "prefer_new": find_prefer_new(t),
    }

    # Validate with the SAME schema /recommend uses
    try:
        profile = RecommendRequest(**data)
    except ValidationError as e:
        raise HTTPException(422, f"Parsed profile was invalid: {e}")

    out = profile.model_dump(exclude={"n_candidates"})
    out["assumed"] = assumed
    return out