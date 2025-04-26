import pandas as pd
from requests import Response

from .explain_utils import get_explanation_contribution_details
from .types import (
    CatDict,
    CountDict,
    ExplainDetailsDict,
    StatsResultsDict,
)


class ExplainResult:
    """ExplainResult class

    Methods
    -------
    These methods are available:
        - `self.flatten_explanation(as_df: bool = False)`
        - `self.get_field_details()`
        - `self.get_field_summary()`

    Attributes
    ----------
    These attributes are available:
        - `status_code`
        - `json`
        - `explanation`
        - `score`
        - `field_details`
        - `field_summary`
    """

    def __init__(self, data: Response | ExplainDetailsDict):
        if isinstance(data, Response):
            self._status_code = data.status_code
            self._response = data
            self._json = data.json()

        elif isinstance(data, dict):  # explanation dict from search results
            if all(key in data for key in ("value", "description", "details")):
                self._status_code = 200
                self._response = None
                self._json = {"explanation": data}

        else:
            raise TypeError("Unsupported input type for ExplainResult")

        self._flatten_explanation = None
        self._field_details = None

    @property
    def status_code(self) -> int:
        """Returns the `status_code`"""
        return self._status_code

    @property
    def json(self) -> dict:
        """Returns the json data"""
        return self._json

    @property
    def explanation(self) -> dict:
        """Returns the explanation"""
        return self.json.get("explanation", {})

    @property
    def score(self) -> float | None:
        """Returns the final score"""
        return self.explanation.get("value")

    @property
    def field_details(self) -> pd.DataFrame:
        """Returns the field/contribution details of the score explanation

        Runs the `get_field_details()` automatically, internally.
        """

        return self.get_field_details()

    @property
    def field_summary(self):
        """Returns the summary of the contribution of each field

        Runs the `get_field_summary()` automatically, internally.
        """

        return self.get_field_summary()

    def flatten_explanation(self, as_df: bool = False) -> list | pd.DataFrame:
        """Gets a breakdown of the scores structure with depth and its information

        Parameters
        ----------
            - as_df (bool): Whether or not to return the results in DataFrame (Default = False)

        Returns
        -------
            list | pd.DataFrame
        """

        if not self._flatten_explanation:
            result = []

            def walk(node, depth=0):
                desc = node.get("description", "")
                val = node.get("value", 0)
                result.append((depth, val, desc))
                for detail in node.get("details", []):
                    walk(detail, depth + 1)

            walk(self.explanation)

            self._flatten_explanation = result

        if as_df:
            return pd.DataFrame(
                self._flatten_explanation, columns=["depth", "score", "description"]
            )

        return self._flatten_explanation

    def get_field_details(self) -> pd.DataFrame:
        """Returns the field/contribution details of the score explanation"""

        if not self._field_details:
            if not self._flatten_explanation:
                self.flatten_explanation()

            results = get_explanation_contribution_details(
                self._flatten_explanation, as_df=True
            )

            self._field_details = results

        return self._field_details

    def get_field_summary(self) -> pd.DataFrame:
        """Returns the summary of the contribution of each field"""

        if not self._field_details:
            self.get_field_details()

        return self._field_details.groupby(["field", "type", "boost"])[["score"]].sum()

    def __repr__(self):
        return f"<ExplainResult explanation_value={self.score}>"


class SearchResults:
    """SearchResults class

    Methods
    -------
    These methods are available:
        - `self.get_sources(include_id: bool = False, include_score: bool = False, as_list: bool = False)`
        - `self.get_ids()`
        - `self.get_explanations()`
        - `self.to_dataframe(columns: list[str], include_id: bool = False, include_score: bool = False)`

    Attributes
    ----------
    These attributes are available:
        - `status_code`
        - `json`
        - `hits`
        - `total`
    """

    def __init__(self, response: Response):
        self._status_code = response.status_code
        self._json: dict = response.json()

    @property
    def status_code(self) -> int:
        """Returns status code from requests"""
        return self._status_code

    @property
    def json(self) -> dict:
        """Returns the json data"""
        return self._json

    @property
    def hits(self) -> list[dict | None]:
        """Returns the `hits` dict of the results"""
        return self.json.get("hits", {}).get("hits", [])

    @property
    def total(self) -> int:
        """Returns a total count"""

        return self.json.get("hits", {}).get("total", {}).get("value", 0)

    def get_total_hits(self):
        """Returns a total count"""
        return self.total

    def get_hits(self) -> list[dict | None]:
        """Returns the `hits` dict of the results"""
        return self.hits

    def get_sources(
        self,
        include_id: bool = False,
        include_score: bool = False,
        as_list: bool = False,
    ) -> dict[str, dict] | list[dict | None]:
        """Returns _source dict results

        Parameters
        ----------
            - include_id (bool): Whether or not to include `_id` in the results

            - include_score (bool): Whether or not to include `_score` in the results

            - as_list (bool): Whether or not to return as a list or dict, default is dict (Default = False)

        Returns
        -------
            dict[str, dict] | list[dict | None]
        """

        if not include_id and not include_score:
            sources_dict = {hit["_id"]: hit["_source"] for hit in self.hits}
            return sources_dict if not as_list else list(sources_dict.values())

        sources_dict = {}
        for hit in self.hits:
            item = hit["_source"].copy()
            if include_id:
                item["_id"] = hit["_id"]
            if include_score:
                item["_score"] = hit["_score"]

            sources_dict[hit["_id"]] = item

        return sources_dict if not as_list else list(sources_dict.values())

    def get_ids(self) -> list[str]:
        """Returns a list of documents' `_id`"""

        return [hit["_id"] for hit in self.hits]

    def get_explanations(self) -> dict[str, ExplainResult] | None:
        """Returns _explanation results

        If `_explanation` exists in the results,
        returns a dictionry of ExplainResult with
        the key being the `_id`

        Example
        -------
            The results will be in a form of:
            ```
            {"xx": <ExplainResult explanation_value=...>,
             "yy": <ExplainResult explanation_value=...>,}
            ```
        """

        if self.hits[0].get("_explanation"):
            return {hit["_id"]: ExplainResult(hit["_explanation"]) for hit in self.hits}

    def to_dataframe(
        self,
        columns: list[str] | None = None,
        include_id: bool = False,
        include_score: bool = False,
    ) -> pd.DataFrame:
        """Returns pandas DataFrame object

        Parameters
        ----------
            - columns (list[str] | None): Columns in the DataFrame results, if None, returns every field (Default = None)

            - include_id (bool): Whether or not to include `_id` in the results

            - include_score (bool): Whether or not to include `_score` in the results

        Returns
        -------
            pd.DataFrame
        """

        df = pd.DataFrame(
            self.get_sources(
                include_id=include_id, include_score=include_score, as_list=False
            )
        ).transpose()

        df.index.name = "_id"

        if columns:
            return df[columns]

        return df

    def __repr__(self):
        return f"<SearchResults total_hits={self.total}>"


class CountResults:
    def __init__(self, response: Response):
        self._status_code = response.status_code
        self._json: CountDict = response.json()

    @property
    def status_code(self) -> int:
        """Returns status code from requests"""
        return self._status_code

    @property
    def json(self) -> CountDict:
        """Returns the json data"""
        return self._json

    @property
    def count(self) -> int:
        """Returns the documents count"""
        return self.json["count"]

    def __repr__(self):
        return f"<CountResults count={self.count}>"


class CatResults:
    def __init__(self, response: Response):
        self._status_code = response.status_code
        self._json: list[dict] = response.json()
        self._json = [self._normalise_cat_keys(row) for row in self._json]

    @property
    def status_code(self) -> int:
        """Returns status code from requests"""
        return self._status_code

    @property
    def json(self) -> list[CatDict]:
        """Returns the json data"""
        return self._json

    @property
    def total(self) -> int:
        """Returns the count"""
        return len(self.json)

    def get_total_indices(self):
        """Returns the total indices count"""
        return self.total

    def get_indices(self) -> list[str]:
        """Returns a list of all the indices"""
        return [row["index"] for row in self.json]

    def filter_indices(self, indices: str | list[str]):
        """Filters the results to the specified index name(s) only"""

        def _filter_results(row):
            if isinstance(indices, str):
                return row["index"] == indices
            elif isinstance(indices, list):
                return row["index"] in indices
            else:
                raise ValueError("indices must be either str or list[str]")

        return list(filter(_filter_results, self.json))

    @staticmethod
    def _normalise_cat_keys(entry: dict) -> CatDict:
        return {k.replace(".", "_"): v for k, v in entry.items()}

    def __repr__(self):
        return f"<CatResults total_indices={self.total}>"


class StatsResults:
    def __init__(self, response: Response):
        self._status_code = response.status_code
        self._json: dict = response.json()

    @property
    def status_code(self) -> int:
        """Returns status code from requests"""
        return self._status_code

    @property
    def json(self) -> StatsResultsDict:
        """Returns the json data"""
        return self._json

    @property
    def total(self):
        """Returns the total documents count"""
        return self.json["_all"]["total"]["docs"]["count"]

    @property
    def size(self):
        """Returns the total size in bytes for the index"""
        return self.json["_all"]["total"]["store"]["size_in_bytes"]

    def get_total_count(self):
        """Returns the total documents count"""
        return self.total

    def get_total_size(self):
        """Returns the total size in bytes for the index"""
        return self.size

    def get_indices(self) -> list[str]:
        """Get a list of indices within this index (maybe aliases)"""
        return list(self.json["indices"].keys())

    def __repr__(self):
        return f"<StatsResults total_count={self.total}, total_size={self.size}>"
