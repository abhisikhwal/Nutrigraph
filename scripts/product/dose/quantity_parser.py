"""Parse recipe ingredient lines into amount, unit, and ingredient name."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

try:
    from ingredient_parser import parse_ingredient as _nlp_parse
except ImportError:
    _nlp_parse = None  # type: ignore

# --- vague / trace quantity policy (documented in dose_layer_report) ---
TRACE_GRAM_DEFAULT = 0.5
PINCH_GRAM = 0.3
DASH_GRAM = 0.4
HANDFUL_GRAM = 30.0
GARNISH_GRAM = 2.0

VAGUE_PATTERNS = re.compile(
    r"\b(to taste|as needed|as desired|for garnish|optional|a pinch|pinch of|"
    r"a dash|dash of|a handful|handful of|few drops|sprinkle)\b",
    re.I,
)

UNIT_ALIASES: dict[str, str] = {
    "c": "cup",
    "c.": "cup",
    "cup": "cup",
    "cups": "cup",
    "tbsp": "tablespoon",
    "tbsp.": "tablespoon",
    "tablespoon": "tablespoon",
    "tablespoons": "tablespoon",
    "tsp": "teaspoon",
    "tsp.": "teaspoon",
    "teaspoon": "teaspoon",
    "teaspoons": "teaspoon",
    "oz": "ounce",
    "oz.": "ounce",
    "ounce": "ounce",
    "ounces": "ounce",
    "lb": "pound",
    "lb.": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "ml": "milliliter",
    "l": "liter",
    "can": "can",
    "cans": "can",
    "clove": "clove",
    "cloves": "clove",
    "slice": "slice",
    "slices": "slice",
    "piece": "piece",
    "pieces": "piece",
    "head": "head",
    "bunch": "bunch",
    "sprig": "sprig",
    "sprigs": "sprig",
    "stick": "stick",
    "sticks": "stick",
    "egg": "egg",
    "eggs": "egg",
    "package": "package",
    "pkg": "package",
}

REGEX_LINE = re.compile(
    r"^\s*"
    r"(?P<amount>(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)(?:\s*-\s*(?:\d+(?:\.\d+)?|\d+/\d+))?)?"
    r"\s*"
    r"(?P<unit>[a-zA-Z][a-zA-Z.\-]*(?:\s+[a-zA-Z.\-]+)?)?"
    r"\s*"
    r"(?P<name>.+?)\s*$",
    re.I,
)


@dataclass
class ParsedQuantity:
    raw: str
    amount: float | None
    unit: str | None
    ingredient_name: str
    parse_class: str  # clean | partial | vague | unparseable
    parser: str  # ingredient_parser | regex | trace_policy
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "amount": self.amount,
            "unit": self.unit,
            "ingredient_name": self.ingredient_name,
            "parse_class": self.parse_class,
            "parser": self.parser,
            "notes": self.notes,
        }


def _frac_to_float(q: Any) -> float | None:
    if q is None:
        return None
    try:
        return float(Fraction(q))
    except (ValueError, TypeError, ZeroDivisionError):
        try:
            return float(q)
        except (ValueError, TypeError):
            return None


def _normalize_unit(u: str | None) -> str | None:
    if not u:
        return None
    key = str(u).strip().lower().rstrip(".")
    return UNIT_ALIASES.get(key, key)


def _unit_str(u: Any) -> str | None:
    if u is None:
        return None
    if hasattr(u, "name"):
        return _normalize_unit(str(u.name))
    return _normalize_unit(str(u))


def _name_from_nlp(parsed: Any) -> str:
    parts = []
    for block in parsed.name or []:
        parts.append(str(block.text))
    return " ".join(parts).strip()


def _parse_nlp(line: str) -> ParsedQuantity | None:
    if _nlp_parse is None:
        return None
    try:
        parsed = _nlp_parse(line)
    except Exception:
        return None
    name = _name_from_nlp(parsed)
    if not name:
        return None
    comment = None
    if parsed.comment:
        comment = str(parsed.comment.text)
    if comment and VAGUE_PATTERNS.search(comment):
        return _apply_trace_policy(line, name, comment)
    if not parsed.amount:
        if re.search(r"\b(eggs?)\b", name, re.I) and re.search(r"^\s*\d", line):
            n = re.match(r"^\s*(\d+(?:\.\d+)?)", line)
            if n:
                return ParsedQuantity(
                    raw=line, amount=float(n.group(1)), unit="egg", ingredient_name=name,
                    parse_class="clean", parser="ingredient_parser",
                    notes=["count unit inferred for eggs"],
                )
        if VAGUE_PATTERNS.search(line):
            return _apply_trace_policy(line, name, line)
        return ParsedQuantity(
            raw=line,
            amount=None,
            unit=None,
            ingredient_name=name,
            parse_class="partial",
            parser="ingredient_parser",
            notes=["name only, no amount"],
        )
    amt = parsed.amount[0]
    amount = _frac_to_float(amt.quantity)
    unit = _unit_str(amt.unit)
    if amount is not None and unit:
        return ParsedQuantity(
            raw=line, amount=amount, unit=unit, ingredient_name=name,
            parse_class="clean", parser="ingredient_parser",
        )
    if amount is not None:
        return ParsedQuantity(
            raw=line, amount=amount, unit=unit, ingredient_name=name,
            parse_class="partial", parser="ingredient_parser",
            notes=["amount without standard unit"],
        )
    return None


def _apply_trace_policy(line: str, name: str, context: str) -> ParsedQuantity:
    ctx = context.lower()
    grams = TRACE_GRAM_DEFAULT
    unit = "trace"
    notes = ["vague quantity → nominal trace amount"]
    if "pinch" in ctx:
        grams, unit, notes = PINCH_GRAM, "pinch", ["pinch → 0.3g nominal"]
    elif "dash" in ctx:
        grams, unit, notes = DASH_GRAM, "dash", ["dash → 0.4g nominal"]
    elif "handful" in ctx:
        grams, unit, notes = HANDFUL_GRAM, "handful", ["handful → 30g nominal"]
    elif "garnish" in ctx:
        grams, unit, notes = GARNISH_GRAM, "garnish", ["garnish → 2g nominal"]
    elif "to taste" in ctx:
        notes = ["to taste → 0.5g nominal trace; flagged not to dominate totals"]
    return ParsedQuantity(
        raw=line,
        amount=grams,
        unit=unit,
        ingredient_name=name,
        parse_class="vague",
        parser="trace_policy",
        notes=notes,
    )


def _parse_regex(line: str) -> ParsedQuantity | None:
    m = REGEX_LINE.match(line.strip())
    if not m:
        return None
    name = (m.group("name") or "").strip(" ,;")
    if not name:
        return None
    amount_raw = m.group("amount")
    unit_raw = m.group("unit")
    amount = None
    if amount_raw:
        amount_raw = amount_raw.strip().replace("-", " ").split()[0]
        amount = _frac_to_float(amount_raw.replace(" ", " "))
    unit = _normalize_unit(unit_raw.split()[0] if unit_raw else None)
    if VAGUE_PATTERNS.search(line):
        return _apply_trace_policy(line, name, line)
    if amount is not None and unit:
        return ParsedQuantity(
            raw=line, amount=amount, unit=unit, ingredient_name=name,
            parse_class="clean", parser="regex",
        )
    if amount is not None:
        return ParsedQuantity(
            raw=line, amount=amount, unit=unit, ingredient_name=name,
            parse_class="partial", parser="regex",
            notes=["amount without unit"],
        )
    return ParsedQuantity(
        raw=line, amount=None, unit=None, ingredient_name=name,
        parse_class="partial", parser="regex",
        notes=["ingredient name only"],
    )


def parse_ingredient_line(line: str) -> ParsedQuantity:
    """Parse one ingredient line; prefer ingredient-parser, fall back to regex."""
    raw = str(line).strip()
    if not raw:
        return ParsedQuantity(
            raw=raw, amount=None, unit=None, ingredient_name="",
            parse_class="unparseable", parser="none", notes=["empty line"],
        )
    if VAGUE_PATTERNS.search(raw) and not re.search(r"\d", raw):
        name = VAGUE_PATTERNS.sub("", raw).strip(" ,-")
        name = re.sub(r"^\W+", "", name) or raw
        return _apply_trace_policy(raw, name, raw)
    nlp = _parse_nlp(raw)
    if nlp:
        if nlp.amount and not nlp.unit and re.search(r"\b(eggs?)\b", nlp.ingredient_name, re.I):
            nlp.unit = "egg"
            nlp.parse_class = "clean"
            nlp.notes.append("count unit inferred for eggs")
        return nlp
    regex = _parse_regex(raw)
    if regex:
        return regex
    return ParsedQuantity(
        raw=raw, amount=None, unit=None, ingredient_name=raw,
        parse_class="unparseable", parser="none",
        notes=["could not extract structure"],
    )
