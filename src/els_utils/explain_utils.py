import math
import re

from .types import ExplainDetailsDict, FieldScoreDict, ScoreSummaryDict


def get_scores_terms(
    dict_or_details_list: list[ExplainDetailsDict] | ExplainDetailsDict,
    results: list = None,
    field_name: str = None,
    clause: str = None,
) -> list[FieldScoreDict]:
    if results is None:
        results = []

    if isinstance(dict_or_details_list, dict):
        value = dict_or_details_list.get("value")
        description = dict_or_details_list.get("description", "")

        # Handle description like 'name.keyword:ปูน*^500.0'
        match = re.match(r"([\w\.]+):(.+?\*\^([\d.]+))", description)
        if match:
            field, clause_str, boost_str = match.groups()
            results.append(
                {
                    "field": field,
                    "clause": clause_str,
                    "type": "value",
                    "value": value,
                }
            )
            results.append(
                {
                    "field": field,
                    "clause": clause_str,
                    "type": "boost",
                    "value": float(boost_str),
                }
            )

        # Set field_name and clause only when pattern is recognized
        if "weight(" in description:
            try:
                segment = description.split("weight(")[1].split(" in ")[0]
                if ":" in segment:
                    field_name, clause = segment.split(":", 1)

            except Exception:
                pass

        elif (
            ":" in description
            and not description.lower().startswith("sum of")
            and not description.lower().startswith("max of")
            and not description.lower().startswith("avg of")
            and not field_name
        ):
            # fallback: try split on `:` if field name wasn't set
            field_name, _, clause = description.partition(":")

        # Match score types
        score_types = ("boost", "idf", "tf")
        for score_type in score_types:
            if description.lower().startswith(score_type):
                results.append(
                    {
                        "field": field_name or "unknown",
                        "clause": clause or "unknown",
                        "type": score_type,
                        "value": value,
                    }
                )

        for detail in dict_or_details_list.get("details", []):
            get_scores_terms(detail, results, field_name, clause)

    elif isinstance(dict_or_details_list, list):
        for item in dict_or_details_list:
            get_scores_terms(item, results, field_name, clause)

    return results


def get_scores_summary(
    details_list: list[FieldScoreDict],
) -> dict[str, ScoreSummaryDict]:
    ret_details = {}
    all_fields = {i["field"] for i in details_list}
    for key in all_fields:
        ret_details[key] = {}
        score_items = [i for i in details_list if i["field"] == key]

        # n_items should either contain 2, 3, or a multiple of 3 items,
        # e.g., [{'field':.., 'type':'boost', 'value':..},
        #        {'field':.., 'type':'tf', 'value':..},
        #        {'field':.., 'type':'idf', 'value':..},
        #       ]

        n_items = len(score_items)
        if n_items <= 2:  # Keyword or exact-match type, store the value and boost as is
            for si in score_items:
                ret_details[key][si["type"]] = si["value"]

        elif n_items == 3:  # boost, tf, idf
            ret_details[key]["value"] = 0
            prod = math.prod([i["value"] for i in score_items])
            ret_details[key]["value"] += prod
            _boost = list(filter(lambda x: x["type"] == "boost", score_items))[
                0
            ]  # get boost dict
            ret_details[key]["boost"] = _boost["value"]

        elif n_items > 3:  # multiple boost, tf, idf for different clauses
            field_clauses = {(i["field"], i["clause"]) for i in score_items}
            ret_details[key]["value"] = 0
            max_boost = max([i["value"] for i in score_items if i["type"] == "boost"])

            for fc_i in field_clauses:
                scores_to_be_multiplied = [
                    i["value"]
                    for i in score_items
                    if i["field"] == fc_i[0] and i["clause"] == fc_i[1]
                ]

                # accumulate for the sub instances, for the same field name, it can have many matches with multiple boost, tf, idf
                # we will capture in a multiple of 3
                n_chunks = len(scores_to_be_multiplied) // 3
                chunk_i = 0
                _sum = 0
                while (
                    chunk_i < n_chunks
                ):  # chunk_i runs from 0, 1, 2 in case len()==8 (3 chunks)
                    _sum += math.prod(
                        scores_to_be_multiplied[chunk_i * 3 : (chunk_i + 1) * 3]
                    )
                    chunk_i += 1

                ret_details[key]["value"] += _sum
                ret_details[key]["boost"] = max_boost

    return ret_details
