from typing import Any

import pandas as pd
from requests import Response


class SearchResults:
    def __init__(self, response: Response):
        self._response = response
        self._json: dict = response.json()

    @property
    def status_code(self) -> int:
        """Returns status code from requests"""
        return self._response.status_code

    @property
    def raw(self) -> Response:
        """Returns raw response from requests"""
        return self._response

    @property
    def hits(self) -> list[dict | None]:
        """Returns the `hits` dict of the results"""
        return self._json.get("hits", {}).get("hits", [])

    @property
    def total(self) -> int:
        """Returns a total count"""

        return self._json.get("hits", {}).get("total", {}).get("value", 0)

    def get_sources(
        self, include_id: bool = False, include_score: bool = False
    ) -> list[dict[str, Any] | None]:
        """Returns a list of _source dict results
        
        Parameters
        ----------
            - include_id (bool): Whether or not to include `_id` in the results

            - include_score (bool): Whether or not to include `_score` in the results

        Returns
        -------
            list[dict | None]
        """

        if not include_id and not include_score:
            return [hit["_source"] for hit in self.hits]

        results = []
        for hit in self.hits:
            item = hit["_source"].copy()
            if include_id:
                item["_id"] = hit["_id"]
            if include_score:
                item["_score"] = hit["_score"]
            results.append(item)
        return results

    def get_ids(self) -> list[str]:
        """Returns a list of documents' `_id`"""

        return [hit["_id"] for hit in self.hits]

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
            self.get_sources(include_id=include_id, include_score=include_score)
        )

        if columns:
            return df[columns]

        return df

    def __repr__(self):
        return f"<SearchResults total_hits={self.total}>"
