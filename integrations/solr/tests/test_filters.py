# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.solr.filters import escape_query_chars, normalize_filters


class TestEscaping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("plain", "plain"),
            ("a+b", "a\\+b"),
            ("(1+1):2", "\\(1\\+1\\)\\:2"),
            ("a b", "a\\ b"),
            ("back\\slash", "back\\\\slash"),
            ('quote"d', 'quote\\"d'),
            ("a&&b", "a\\&\\&b"),
        ],
    )
    def test_escape_query_chars(self, raw, expected):
        assert escape_query_chars(raw) == expected


class TestComparisons:
    def test_equal_picks_the_field_matching_the_value_type(self):
        """Strings, bools stay in their own field; integers query both plong and pdouble."""
        assert normalize_filters({"field": "meta.page", "operator": "==", "value": "100"}) == 'meta_s_page:"100"'
        assert normalize_filters({"field": "meta.page", "operator": "==", "value": True}) == "meta_b_page:true"
        # Integer 100 could be stored as plong (100) or pdouble (100.0) — query both.
        clause = normalize_filters({"field": "meta.page", "operator": "==", "value": 100})
        assert "meta_l_page:100" in clause
        assert "meta_d_page:100.0" in clause
        # Non-integer float has no integer equivalent, so only the pdouble field is queried.
        assert normalize_filters({"field": "meta.page", "operator": "==", "value": 1.5}) == "meta_d_page:1.5"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # Non-integer floats only match the pdouble field.
            (2.5, "meta_d_n:2.5"),
            (-2.5, "meta_d_n:\\-2.5"),
        ],
    )
    def test_negative_numbers_are_escaped_non_integer(self, value, expected):
        """Non-integer float equality only queries the pdouble field."""
        assert normalize_filters({"field": "meta.n", "operator": "==", "value": value}) == expected

    @pytest.mark.parametrize("value", [10, -10])
    def test_negative_numbers_are_escaped_integer(self, value):
        """
        An unescaped leading minus is a Lucene syntax error, and inside a group it silently negates.

        Integer equality queries both plong and pdouble; both fields escape the leading minus.
        """
        clause = normalize_filters({"field": "meta.n", "operator": "==", "value": value})
        fv = float(value)
        escaped_int = f"\\{value}" if value < 0 else str(value)
        escaped_float = f"\\{fv}" if fv < 0 else str(fv)
        assert f"meta_l_n:{escaped_int}" in clause
        assert f"meta_d_n:{escaped_float}" in clause

    def test_negative_numbers_are_escaped_in_groups(self):
        clause = normalize_filters({"field": "meta.n", "operator": "in", "value": [-10, 100]})
        assert "meta_l_n:(\\-10 OR 100)" in clause
        assert "meta_d_n:(\\-10.0 OR 100.0)" in clause

    def test_negative_numbers_are_escaped_in_ranges(self):
        clause = normalize_filters({"field": "meta.n", "operator": ">=", "value": -10})
        assert "meta_l_n:[\\-10 TO *]" in clause
        assert "meta_d_n:[\\-10 TO *]" in clause

    def test_meta_prefix_is_optional(self):
        with_prefix = normalize_filters({"field": "meta.page", "operator": "==", "value": "x"})
        without_prefix = normalize_filters({"field": "page", "operator": "==", "value": "x"})
        assert with_prefix == without_prefix

    def test_top_level_fields_carry_no_type_code(self):
        assert normalize_filters({"field": "id", "operator": "==", "value": "1"}) == 'id:"1"'
        assert normalize_filters({"field": "content", "operator": "==", "value": "hi"}) == 'content:"hi"'

    def test_equal_none_means_the_key_is_absent(self):
        clause = normalize_filters({"field": "meta.number", "operator": "==", "value": None})
        assert clause.startswith("(*:* -(")
        assert "meta_l_number:[* TO *]" in clause
        assert "meta_s_number:[* TO *]" in clause

    def test_not_equal_none_means_the_key_is_present(self):
        clause = normalize_filters({"field": "meta.number", "operator": "!=", "value": None})
        assert not clause.startswith("(*:* -(")
        assert "meta_l_number:[* TO *]" in clause

    def test_not_equal_also_matches_documents_missing_the_field(self):
        """`None != 100` is true in Python, so a Solr negation is the right translation."""
        clause = normalize_filters({"field": "meta.number", "operator": "!=", "value": 100})
        # Negation wraps the cross-type equality (both plong and pdouble).
        assert "meta_l_number:100" in clause
        assert "meta_d_number:100.0" in clause
        assert clause.startswith("(*:* -(")

    @pytest.mark.parametrize(
        ("operator", "l_expected", "d_expected"),
        [
            # For int value 100: plong uses closed brackets with the shifted boundary;
            # pdouble uses the original open/closed template.
            (">", "meta_l_number:[101 TO *]", "meta_d_number:{100 TO *}"),
            (">=", "meta_l_number:[100 TO *]", "meta_d_number:[100 TO *]"),
            ("<", "meta_l_number:[* TO 99]", "meta_d_number:[* TO 100}"),
            ("<=", "meta_l_number:[* TO 100]", "meta_d_number:[* TO 100]"),
        ],
    )
    def test_ranges(self, operator, l_expected, d_expected):
        clause = normalize_filters({"field": "meta.number", "operator": operator, "value": 100})
        assert l_expected in clause
        assert d_expected in clause

    def test_iso_dates_compare_lexicographically_on_a_string_field(self):
        clause = normalize_filters({"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"})
        assert clause == 'meta_s_date:{"1972\\-12\\-11T19\\:54\\:58" TO *}'

    @pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
    def test_non_date_strings_are_rejected(self, operator):
        with pytest.raises(FilterError, match="Strings must be ISO formatted dates"):
            normalize_filters({"field": "meta.date", "operator": operator, "value": "not-a-date"})

    def test_in_with_only_nulls_means_the_key_is_absent(self):
        clause = normalize_filters({"field": "meta.n", "operator": "in", "value": [None]})
        assert clause.startswith("(*:* -(")
        assert "meta_l_n:[* TO *]" in clause

    def test_in_mixing_nulls_and_values_covers_both(self):
        clause = normalize_filters({"field": "meta.n", "operator": "in", "value": [1, None]})
        assert "meta_l_n:(1)" in clause
        assert "-(" in clause

    def test_existence_of_a_top_level_field_needs_no_type_code(self):
        assert normalize_filters({"field": "id", "operator": "!=", "value": None}) == "id:[* TO *]"

    def test_unknown_operator(self):
        with pytest.raises(FilterError, match="Unknown operator"):
            normalize_filters({"field": "meta.number", "operator": "~=", "value": 1})


class TestNumericCrossTypeFilters:
    """
    Numeric metadata may be stored in either the plong (int) or pdouble (float) Solr field.

    A filter such as ``rating >= 4`` must search *both* ``meta_l_rating`` and ``meta_d_rating``
    so that documents stored with int ``4`` and documents stored with float ``4.5`` are both
    returned.  See https://github.com/deepset-ai/haystack-core-integrations/issues/4083.
    """

    def test_range_ge_int_queries_both_type_fields(self):
        """``>= 4`` (int) must cover plong-stored 4 and pdouble-stored 4.5, 5.0, …"""
        clause = normalize_filters({"field": "meta.rating", "operator": ">=", "value": 4})
        assert "meta_l_rating:[4 TO *]" in clause
        assert "meta_d_rating:[4 TO *]" in clause

    def test_range_ge_float_queries_both_type_fields(self):
        """``>= 4.0`` (float) must cover both storage types."""
        clause = normalize_filters({"field": "meta.rating", "operator": ">=", "value": 4.0})
        assert "meta_l_rating:[4 TO *]" in clause
        assert "meta_d_rating:[4.0 TO *]" in clause

    def test_range_float_boundary_rounds_int_threshold_up(self):
        """``>= 4.2`` on the plong field becomes ``[5 TO *]`` (first int >= 4.2 is 5)."""
        clause = normalize_filters({"field": "meta.rating", "operator": ">=", "value": 4.2})
        assert "meta_l_rating:[5 TO *]" in clause
        assert "meta_d_rating:[4.2 TO *]" in clause

    def test_range_gt_int_shifts_plong_boundary(self):
        """``> 4`` on plong becomes ``[5 TO *]`` (5 is the first int strictly greater than 4)."""
        clause = normalize_filters({"field": "meta.rating", "operator": ">", "value": 4})
        assert "meta_l_rating:[5 TO *]" in clause
        assert "meta_d_rating:{4 TO *}" in clause

    def test_range_lt_float_rounds_plong_boundary_down(self):
        """``< 4.2`` on plong becomes ``[* TO 4]`` (4 is the last int strictly less than 4.2)."""
        clause = normalize_filters({"field": "meta.rating", "operator": "<", "value": 4.2})
        assert "meta_l_rating:[* TO 4]" in clause
        assert "meta_d_rating:[* TO 4.2}" in clause

    def test_range_lt_integer_value_shifts_plong_boundary(self):
        """``< 4`` on plong becomes ``[* TO 3]`` (last int < 4 is 3)."""
        clause = normalize_filters({"field": "meta.rating", "operator": "<", "value": 4})
        assert "meta_l_rating:[* TO 3]" in clause
        assert "meta_d_rating:[* TO 4}" in clause

    def test_range_le_float_queries_both_type_fields(self):
        """``<= 4.2`` on plong becomes ``[* TO 4]`` (floor of 4.2)."""
        clause = normalize_filters({"field": "meta.rating", "operator": "<=", "value": 4.2})
        assert "meta_l_rating:[* TO 4]" in clause
        assert "meta_d_rating:[* TO 4.2]" in clause

    def test_equality_int_also_matches_float_field(self):
        """``score == 5`` must also match documents where ``score`` was stored as the float 5.0."""
        clause = normalize_filters({"field": "meta.score", "operator": "==", "value": 5})
        assert "meta_l_score:5" in clause
        assert "meta_d_score:5.0" in clause

    def test_equality_whole_float_also_matches_int_field(self):
        """``score == 5.0`` must also match documents where ``score`` was stored as the int 5."""
        clause = normalize_filters({"field": "meta.score", "operator": "==", "value": 5.0})
        assert "meta_l_score:5" in clause
        assert "meta_d_score:5.0" in clause

    def test_equality_non_integer_float_only_matches_float_field(self):
        """``score == 4.5`` cannot match any integer, so only the pdouble field is queried."""
        clause = normalize_filters({"field": "meta.score", "operator": "==", "value": 4.5})
        assert "meta_d_score:4.5" in clause
        assert "meta_l_score" not in clause

    def test_in_int_list_queries_both_type_fields(self):
        """``rating in [4, 5]`` must find int-stored 4 and float-stored 5.0."""
        clause = normalize_filters({"field": "meta.rating", "operator": "in", "value": [4, 5]})
        assert "meta_l_rating" in clause
        assert "meta_d_rating" in clause

    def test_in_float_list_queries_both_type_fields_for_whole_numbers(self):
        """``rating in [4.0, 5.0]`` must find int-stored 4 and int-stored 5."""
        clause = normalize_filters({"field": "meta.rating", "operator": "in", "value": [4.0, 5.0]})
        assert "meta_l_rating" in clause
        assert "meta_d_rating" in clause

    def test_in_non_integer_float_only_uses_float_field(self):
        """``rating in [4.5]`` cannot match any integer so only pdouble is queried."""
        clause = normalize_filters({"field": "meta.rating", "operator": "in", "value": [4.5]})
        assert "meta_d_rating:(4.5)" in clause
        assert "meta_l_rating" not in clause

    def test_not_in_numeric_also_covers_both_type_fields(self):
        """``rating not in [4, 5]`` is the negation of the cross-type in-clause."""
        clause = normalize_filters({"field": "meta.rating", "operator": "not in", "value": [4, 5]})
        assert "meta_l_rating" in clause
        assert "meta_d_rating" in clause
        assert clause.startswith("(*:* -(")


class TestLogicalConditions:
    def test_and(self):
        clause = normalize_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.page", "operator": "==", "value": "100"},
                    {"field": "meta.number", "operator": ">", "value": 1},
                ],
            }
        )
        assert 'meta_s_page:"100"' in clause
        assert "meta_l_number:[2 TO *]" in clause
        assert "meta_d_number:{1 TO *}" in clause

    def test_or(self):
        clause = normalize_filters(
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.page", "operator": "==", "value": "100"},
                    {"field": "meta.page", "operator": "==", "value": "90"},
                ],
            }
        )
        assert clause == '(meta_s_page:"100" OR meta_s_page:"90")'

    def test_not(self):
        clause = normalize_filters(
            {"operator": "NOT", "conditions": [{"field": "meta.page", "operator": "==", "value": "100"}]}
        )
        assert clause == '(*:* -(meta_s_page:"100"))'

    def test_nesting(self):
        clause = normalize_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.number", "operator": "==", "value": 2},
                            {"field": "meta.number", "operator": "==", "value": -10},
                        ],
                    },
                ],
            }
        )
        assert 'meta_s_chapter:"intro"' in clause
        assert "meta_l_number:2" in clause
        assert "meta_d_number:2.0" in clause
        assert "meta_l_number:\\-10" in clause
        assert "meta_d_number:\\-10.0" in clause

    def test_unknown_logical_operator(self):
        with pytest.raises(FilterError, match="Unknown logical operator"):
            normalize_filters({"operator": "XOR", "conditions": []})


class TestMalformedFilters:
    def test_non_dict(self):
        with pytest.raises(FilterError, match="Filters must be a dictionary"):
            normalize_filters(["not", "a", "dict"])

    def test_missing_operator_key(self):
        with pytest.raises(FilterError, match="'operator' key missing"):
            normalize_filters({"conditions": []})

    def test_missing_conditions_key(self):
        with pytest.raises(FilterError, match="'conditions' key missing"):
            normalize_filters({"operator": "AND"})

    def test_missing_value_key(self):
        with pytest.raises(FilterError, match="'value' key missing"):
            normalize_filters({"field": "meta.page", "operator": "=="})

    def test_missing_condition_operator_key(self):
        with pytest.raises(FilterError, match="'operator' key missing"):
            normalize_filters({"field": "meta.page", "value": "100"})

    def test_explicit_none_field(self):
        with pytest.raises(FilterError, match="'field' key missing"):
            normalize_filters({"field": None, "operator": "==", "value": "100"})

    def test_empty_conditions_match_everything(self):
        assert normalize_filters({"operator": "AND", "conditions": []}) == "*:*"
