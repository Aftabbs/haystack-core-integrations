# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Translation of Haystack filters into Solr filter query (`fq`) clauses."""

import math
from datetime import datetime
from typing import Any

from haystack.errors import FilterError

from .schema import ALL_TYPE_CODES, CONTENT_FIELD, ID_FIELD, meta_field_name, type_code_for_value

#: Characters the Lucene query parser treats as syntax. Mirrors SolrJ's `ClientUtils.escapeQueryChars`.
_SPECIAL_CHARACTERS = set('\\+-!():^[]"{}~*?|&;/')

#: Top-level Solr fields that are not metadata and therefore carry no type code.
_SPECIAL_FIELDS = {ID_FIELD, CONTENT_FIELD}

#: Matches every document. Used to turn a bare negation into a well-formed clause.
_MATCH_ALL = "*:*"

#: Matches nothing. Used where a comparison is satisfiable by no document.
_MATCH_NONE = f"(-{_MATCH_ALL})"

_RANGE_TEMPLATES = {
    ">": "{{{value} TO *}}",
    ">=": "[{value} TO *]",
    "<": "[* TO {value}}}",
    "<=": "[* TO {value}]",
}

# Range templates for the plong field when querying with a numeric value.
# The int boundary is always exact, so closed brackets are used on both sides.
_INT_RANGE_FORMATS = {
    ">=": "[{value} TO *]",
    ">": "[{value} TO *]",   # boundary is already shifted to next int
    "<=": "[* TO {value}]",
    "<": "[* TO {value}]",   # boundary is already shifted to previous int
}


def escape_query_chars(value: str) -> str:
    """
    Escape the Lucene syntax characters in `value`.

    :param value: the raw string.
    :returns: the string with every syntax character and every whitespace run backslash-escaped.
    """
    return "".join(f"\\{char}" if char in _SPECIAL_CHARACTERS or char.isspace() else char for char in value)


def _format_value(value: Any) -> str:
    """Render a scalar as a Solr query term."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # A leading minus opens a prohibited clause in the Lucene parser, which makes `field:-10` an
        # outright syntax error and, worse, silently turns `field:(-10 OR 100)` into "100 but not 10".
        # Escaping it keeps the value a term in every position: equality, ranges and grouped clauses.
        return f"\\{value}" if value < 0 else str(value)
    return f'"{escape_query_chars(str(value))}"'


def _normalize_field_name(field: str) -> str:
    """Strip the `meta.` prefix Haystack filters use to address metadata."""
    return field[5:] if field.startswith("meta.") else field


def _resolve_field(field: str, value: Any) -> str:
    """
    Map a Haystack filter field onto the Solr field holding values of `value`'s type.

    :param field: the Haystack field name, with or without a `meta.` prefix.
    :param value: the value being compared, whose Python type selects the type code.
    :returns: the Solr field name.
    """
    normalized = _normalize_field_name(field)
    if normalized in _SPECIAL_FIELDS:
        return normalized
    return meta_field_name(normalized, type_code_for_value(value))


def _is_numeric(value: Any) -> bool:
    """Return True for int or float values (bool is excluded because it subclasses int)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_range_boundary(value: float, operator: str) -> int:
    """
    Convert a float filter value to the equivalent closed-bracket integer boundary.

    Solr's plong field rejects float literals, so range queries must use an integer bound.
    The int boundary is shifted so that the closed-bracket template produces the same
    set of matching integers as the original operator would.

    Examples::

        >= 4.2  ->  5   ([5 TO *]  covers all ints >= 5, i.e. > 4.2)
        >  4.0  ->  5   ([5 TO *]  covers all ints > 4.0)
        <  4.2  ->  4   ([* TO 4]  covers all ints < 4.2)
        <= 4.0  ->  4   ([* TO 4]  covers all ints <= 4.0)
    """
    fv = float(value)
    if operator == ">=":
        return math.ceil(fv)
    if operator == ">":
        return math.floor(fv) + 1
    if operator == "<=":
        return math.floor(fv)
    # operator == "<"
    return math.ceil(fv) - 1


def _negate(clause: str) -> str:
    """
    Wrap `clause` in a negation that is valid in every position.

    A leading-negative clause is accepted as a whole `fq` but not as a nested boolean operand, so the
    negation is always paired with an explicit `*:*`.
    """
    return f"({_MATCH_ALL} -({clause}))"


def _existence_clause(field: str) -> str:
    """
    Build a clause matching documents where `field` has a value under any type code.

    Because the type code is part of the field name, "the metadata key exists" is the union over every
    field that key could have been stored in.
    """
    normalized = _normalize_field_name(field)
    if normalized in _SPECIAL_FIELDS:
        return f"{normalized}:[* TO *]"
    union = " OR ".join(f"{meta_field_name(normalized, code)}:[* TO *]" for code in ALL_TYPE_CODES)
    return f"({union})"


def _equality_clause(field: str, value: Any) -> str:
    """Build an `==` clause."""
    if value is None:
        # `== None` means the key is absent, since a null metadata value is never written.
        return _negate(_existence_clause(field))
    if isinstance(value, list):
        # Matching a whole list means matching every element of the multi-valued field.
        if not value:
            return _negate(_existence_clause(field))
        solr_field = _resolve_field(field, value)
        return "(" + " AND ".join(f"{solr_field}:{_format_value(element)}" for element in value) + ")"
    normalized = _normalize_field_name(field)
    if _is_numeric(value) and normalized not in _SPECIAL_FIELDS:
        # Numeric metadata may be stored in either the plong (int) or pdouble (float) field.
        # Query both so that e.g. `score == 5` also matches documents where `score` was stored
        # as the float 5.0, and `score == 5.0` also matches an int-stored 5.
        d_field = meta_field_name(normalized, "d")
        float_val = float(value)
        d_clause = f"{d_field}:{_format_value(float_val)}"
        if float_val == math.floor(float_val):
            # Whole number: could also be stored as a plong.
            int_val = int(float_val)
            l_field = meta_field_name(normalized, "l")
            l_clause = f"{l_field}:{_format_value(int_val)}"
            return f"({l_clause} OR {d_clause})"
        return d_clause
    return f"{_resolve_field(field, value)}:{_format_value(value)}"


def _range_clause(field: str, operator: str, value: Any) -> str:
    """Build a `>`, `>=`, `<` or `<=` clause."""
    if value is None:
        # No document compares to null, matching how the other document stores behave.
        return _MATCH_NONE
    if isinstance(value, list):
        msg = f"Filter value can't be of type list using operator {operator!r}"
        raise FilterError(msg)
    if isinstance(value, str):
        # Solr compares strings lexicographically, which is only meaningful for ISO-8601 timestamps.
        try:
            datetime.fromisoformat(value)
        except (TypeError, ValueError) as error:
            msg = f"Can't compare strings using operator {operator!r}. Strings must be ISO formatted dates."
            raise FilterError(msg) from error
    if isinstance(value, bool):
        msg = f"Filter value can't be of type bool using operator {operator!r}"
        raise FilterError(msg)

    normalized = _normalize_field_name(field)
    if _is_numeric(value) and normalized not in _SPECIAL_FIELDS:
        # Numeric metadata may be stored in either the plong (int) or pdouble (float) field.
        # Query both so that e.g. `rating >= 4` also matches documents with float-stored 4.5,
        # and `rating < 4.2` also matches documents with int-stored 3 or 4.
        #
        # Solr's plong field rejects float literals in range boundaries, so the int boundary is
        # adjusted to the nearest integer that produces the same set of matching longs.
        int_boundary = _int_range_boundary(float(value), operator)
        l_field = meta_field_name(normalized, "l")
        d_field = meta_field_name(normalized, "d")
        l_clause = f"{l_field}:{_INT_RANGE_FORMATS[operator].format(value=_format_value(int_boundary))}"
        d_clause = f"{d_field}:{_RANGE_TEMPLATES[operator].format(value=_format_value(value))}"
        return f"({l_clause} OR {d_clause})"

    solr_field = _resolve_field(field, value)
    return f"{solr_field}:{_RANGE_TEMPLATES[operator].format(value=_format_value(value))}"


def _in_clause(field: str, operator: str, value: Any) -> str:
    """Build an `in` clause. Values are grouped by type code, since each type lives in its own field."""
    if not isinstance(value, list):
        msg = f"{operator!r} operator must have a list of values as filter value, but got {type(value)}"
        raise FilterError(msg)
    if not value:
        return _MATCH_NONE

    normalized = _normalize_field_name(field)
    grouped: dict[str, list[Any]] = {}
    for element in value:
        if element is None:
            continue
        if _is_numeric(element) and normalized not in _SPECIAL_FIELDS:
            # Numeric elements may be stored in either the plong or pdouble field.
            # Add each element to both groups so that an int filter value also matches float-stored
            # equivalents and vice versa (e.g. `in [4, 5]` finds both int-stored 4 and float-stored 5.0).
            float_val = float(element)
            if float_val == math.floor(float_val):
                # Whole number: include in the plong group.
                int_val = int(float_val)
                l_field = meta_field_name(normalized, "l")
                grouped.setdefault(l_field, []).append(int_val)
            d_field = meta_field_name(normalized, "d")
            grouped.setdefault(d_field, []).append(float_val)
        else:
            grouped.setdefault(_resolve_field(field, element), []).append(element)
    if not grouped:
        # A list of nothing but nulls can only match documents that have no value for the field.
        return _negate(_existence_clause(field))

    clauses = [
        f"{solr_field}:({' OR '.join(_format_value(element) for element in elements)})"
        for solr_field, elements in grouped.items()
    ]
    if None in value:
        clauses.append(_negate(_existence_clause(field)))
    return "(" + " OR ".join(clauses) + ")"


def _parse_comparison_condition(condition: dict[str, Any]) -> str:
    """Translate a single comparison condition."""
    field = condition.get("field")
    if field is None:
        msg = f"'field' key missing in {condition}"
        raise FilterError(msg)
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    value = condition["value"]

    if operator == "==":
        return _equality_clause(field, value)
    if operator == "!=":
        if value is None:
            # `!= None` means the key is present.
            return _existence_clause(field)
        # A Solr negation also matches documents missing the field, which is what Haystack expects:
        # `None != 100` is true, so documents without the key belong in the result.
        return _negate(_equality_clause(field, value))
    if operator in _RANGE_TEMPLATES:
        return _range_clause(field, operator, value)
    if operator == "in":
        return _in_clause(field, operator, value)
    if operator == "not in":
        return _negate(_in_clause(field, operator, value))

    msg = f"Unknown operator {operator}"
    raise FilterError(msg)


def _parse_logical_condition(condition: dict[str, Any]) -> str:
    """Translate a logical condition and its nested conditions."""
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ("AND", "OR", "NOT"):
        # Validated before the conditions, so that an unknown operator is reported even when the
        # condition list is empty.
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)

    conditions = [_parse_condition(nested) for nested in condition["conditions"]]
    if not conditions:
        return _MATCH_ALL

    if operator == "AND":
        return "(" + " AND ".join(conditions) + ")"
    if operator == "OR":
        return "(" + " OR ".join(conditions) + ")"
    return _negate(" AND ".join(conditions))


def _parse_condition(condition: dict[str, Any]) -> str:
    """Translate either a comparison or a logical condition."""
    if "field" in condition:
        return _parse_comparison_condition(condition)
    return _parse_logical_condition(condition)


def normalize_filters(filters: dict[str, Any]) -> str:
    """
    Convert Haystack filters into a single Solr filter query clause.

    :param filters: the filters to convert, in Haystack's comparison/logic dictionary form.
    :returns: a clause suitable for Solr's `fq` parameter or for a delete-by-query.
    :raises FilterError: if `filters` is malformed or uses an unsupported operator or value type.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)
    return _parse_condition(filters)
