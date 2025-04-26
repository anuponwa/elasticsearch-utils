import pandas as pd

# from .types import ExplainDetailsDict, FieldScoreDict, ScoreSummaryDict


# def get_scores_terms(
#     dict_or_details_list: list[ExplainDetailsDict] | ExplainDetailsDict,
#     results: list = None,
#     field_name: str = None,
#     clause: str = None,
# ) -> list[FieldScoreDict]:
#     if results is None:
#         results = []

#     if isinstance(dict_or_details_list, dict):
#         value = dict_or_details_list.get("value")
#         description = dict_or_details_list.get("description", "")

#         # Handle description like 'name.keyword:ปูน*^500.0'
#         match = re.match(r"([\w\.]+):(.+?\*\^([\d.]+))", description)
#         if match:
#             field, clause_str, boost_str = match.groups()
#             results.append(
#                 {
#                     "field": field,
#                     "clause": clause_str,
#                     "type": "value",
#                     "value": value,
#                 }
#             )
#             results.append(
#                 {
#                     "field": field,
#                     "clause": clause_str,
#                     "type": "boost",
#                     "value": float(boost_str),
#                 }
#             )

#         # Set field_name and clause only when pattern is recognized
#         if "weight(" in description:
#             try:
#                 segment = description.split("weight(")[1].split(" in ")[0]
#                 if ":" in segment:
#                     field_name, clause = segment.split(":", 1)

#             except Exception:
#                 pass

#         elif (
#             ":" in description
#             and not description.lower().startswith("sum of")
#             and not description.lower().startswith("max of")
#             and not description.lower().startswith("avg of")
#             and not field_name
#         ):
#             # fallback: try split on `:` if field name wasn't set
#             field_name, _, clause = description.partition(":")

#         # Match score types
#         score_types = ("boost", "idf", "tf")
#         for score_type in score_types:
#             if description.lower().startswith(score_type):
#                 results.append(
#                     {
#                         "field": field_name or "unknown",
#                         "clause": clause or "unknown",
#                         "type": score_type,
#                         "value": value,
#                     }
#                 )

#         for detail in dict_or_details_list.get("details", []):
#             get_scores_terms(detail, results, field_name, clause)

#     elif isinstance(dict_or_details_list, list):
#         for item in dict_or_details_list:
#             get_scores_terms(item, results, field_name, clause)

#     return results


# def get_scores_summary(
#     details_list: list[FieldScoreDict],
# ) -> dict[str, ScoreSummaryDict]:
#     ret_details = {}
#     all_fields = {i["field"] for i in details_list}
#     for key in all_fields:
#         ret_details[key] = {}
#         score_items = [i for i in details_list if i["field"] == key]

#         # n_items should either contain 2, 3, or a multiple of 3 items,
#         # e.g., [{'field':.., 'type':'boost', 'value':..},
#         #        {'field':.., 'type':'tf', 'value':..},
#         #        {'field':.., 'type':'idf', 'value':..},
#         #       ]

#         n_items = len(score_items)
#         if n_items <= 2:  # Keyword or exact-match type, store the value and boost as is
#             for si in score_items:
#                 ret_details[key][si["type"]] = si["value"]

#         elif n_items == 3:  # boost, tf, idf
#             ret_details[key]["value"] = 0
#             prod = math.prod([i["value"] for i in score_items])
#             ret_details[key]["value"] += prod
#             _boost = list(filter(lambda x: x["type"] == "boost", score_items))[
#                 0
#             ]  # get boost dict
#             ret_details[key]["boost"] = _boost["value"]

#         elif n_items > 3:  # multiple boost, tf, idf for different clauses
#             field_clauses = {(i["field"], i["clause"]) for i in score_items}
#             ret_details[key]["value"] = 0
#             max_boost = max([i["value"] for i in score_items if i["type"] == "boost"])

#             for fc_i in field_clauses:
#                 scores_to_be_multiplied = [
#                     i["value"]
#                     for i in score_items
#                     if i["field"] == fc_i[0] and i["clause"] == fc_i[1]
#                 ]

#                 # accumulate for the sub instances, for the same field name, it can have many matches with multiple boost, tf, idf
#                 # we will capture in a multiple of 3
#                 n_chunks = len(scores_to_be_multiplied) // 3
#                 chunk_i = 0
#                 _sum = 0
#                 while (
#                     chunk_i < n_chunks
#                 ):  # chunk_i runs from 0, 1, 2 in case len()==8 (3 chunks)
#                     _sum += math.prod(
#                         scores_to_be_multiplied[chunk_i * 3 : (chunk_i + 1) * 3]
#                     )
#                     chunk_i += 1

#                 ret_details[key]["value"] += _sum
#                 ret_details[key]["boost"] = max_boost

#     return ret_details


# New extraction functions


def get_field_and_term_from_main(text_details: str) -> tuple[str, str]:
    """Get the field and matched term from the main weight() tf*idf score"""

    text_split = text_details.split("weight(")[1].split(":")
    field = text_split[0]
    term = text_split[1].split(" in")[0]
    return field, term


def get_field_and_term_from_boost(text_details: str) -> tuple[str, str]:
    """Get the field, term and the boost from keywork/boost score type"""

    text_split = text_details.split(":")
    field = text_split[0]
    term_boost = text_split[1].split("*^")
    term = term_boost[0]
    boost = term_boost[1]
    return field, term, boost


def get_field_and_terms_from_synonym(text_details: str) -> tuple[str, list[str]]:
    """Get the field and matched terms from the synonym score"""

    inner = text_details.split("Synonym(")[1].split(")")[0]
    parts = inner.strip().split()
    field = parts[0].split(":")[0]
    terms = []
    terms.append(parts[0].split(":")[1])

    for p in parts:
        if ":" not in p:
            if p not in terms:
                terms.append(p)

    return field, "|".join(terms)


def get_explanation_contribution_details(
    break_down: list[tuple[int, float, str]], as_df: bool = False
) -> list[dict] | pd.DataFrame:
    """Returns the contribution details of each field from the flatten explanation
    
    Parameters
    ----------
    break_down: list[tuple[int, float, str]]
        The break-down, flatten explanation structure that has:

        `[depth, score, details]` format.

    as_df: bool (Default = False)
        Whether or not to return this as a DataFrame

    Returns
    -------
    list[dict] | pd.DataFrame
        A detailed contribution structure, for example:

        ```
        [
            {
                "id": 1,
                "type": "Weight",
                "field": "name",
                "term": "some-product",
                "score": 314.159,
                "boost": 2.2,
                "depth": 3,
                "op": None,
                "parent_depth": 1,
                "parent_id": 1,
                "parent_op": "sum"
            }
        ]
        ```
    """

    results = []
    for i, (depth, score, details) in enumerate(break_down):
        if details in ("sum of:", "avg of:", "max of:", "min of:"):
            cur_depth = depth  # Set the current depth level
            cur_op = details.split()[0]
            cur_id = i

            parent_depth = 0
            parent_id = 0
            parent_op = None

            for k in range(0, i):
                _parent_depth = break_down[k][0]
                parent_details = break_down[k][2]
                if _parent_depth == depth - 1 and parent_details in (
                    "sum of:",
                    "avg of:",
                    "max of:",
                    "min of:",
                ):
                    parent_depth = _parent_depth
                    parent_id = k
                    parent_op = parent_details.split()[0]

            contrib_dict = {
                "id": i,
                "type": "Aggregate",
                "field": None,
                "term": None,
                "score": None,
                "boost": None,
                "depth": depth,
                "op": cur_op,
                "parent_depth": parent_depth,
                "parent_id": parent_id,
                "parent_op": parent_op,
            }

            results.append(contrib_dict)
            # continue  # Move on to its details

        for i_ in range(i + 1, len(break_down)):
            if i_ in [r["id"] for r in results]:
                continue  # Do not append the results that are already in the list, avoid duplicates

            child_depth = break_down[i_][0]
            child_score = break_down[i_][1]
            child_details = break_down[i_][2]

            if (
                child_depth == cur_depth + 1
            ):  # one level deeper than the current aggregate
                if child_details.startswith("weight("):
                    if "Synonym" in child_details:
                        field, term = get_field_and_terms_from_synonym(child_details)
                        type_ = "Synonym"
                    else:
                        field, term = get_field_and_term_from_main(child_details)
                        type_ = "Weight"

                    for j in range(i_, len(break_down)):
                        dtl_depth = break_down[j][0]
                        dtl_score = break_down[j][1]
                        dtl_detail = break_down[j][2]

                        if (
                            dtl_depth == child_depth + 2 and dtl_detail == "boost"
                        ):  ## two level deeper than the main score
                            boost = dtl_score

                    contrib_dict = {
                        "id": i_,
                        "type": type_,
                        "field": field,
                        "term": term,
                        "score": child_score,
                        "boost": boost,
                        "depth": child_depth,
                        "op": None,
                        "parent_depth": cur_depth,
                        "parent_id": cur_id,
                        "parent_op": cur_op,
                    }
                    results.append(contrib_dict)

                if "*^" in child_details:
                    field, term, boost = get_field_and_term_from_boost(child_details)

                    contrib_dict = {
                        "id": i_,
                        "type": "Keyword",
                        "field": field,
                        "term": term,
                        "score": boost,
                        "boost": boost,
                        "depth": child_depth,
                        "op": None,
                        "parent_depth": cur_depth,
                        "parent_id": cur_id,
                        "parent_op": cur_op,
                    }
                    results.append(contrib_dict)

    if as_df:
        df_results = pd.DataFrame(results)
        df_results["score"] = df_results["score"].astype(float)

        return df_results

    return results
