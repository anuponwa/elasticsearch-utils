import base64
import json
import posixpath as ppath
from typing import Literal

import requests

from .response import RequestResponse
from .results import (
    CatResults,
    CountResults,
    SearchResults,
    ExplainResult,
    StatsResults,
)


class ESClient:
    def __init__(
        self,
        es_endpoint: str | None = None,
        api_key: str | None = None,
        basic_auth: tuple[str, str] | None = None,
    ):
        """Initialise Elasticsearch tool object

        Parameters
        ----------
            - es_endpoint (str): URL endpoint of the Elasticsearch cluster

            - api_key (str): API key for authentication (for API key authentication)

            - basic_auth (tuple[str, str]): A tuple of (username, password) (for basic authentication)
        """
        self.es_endpoint = es_endpoint
        self.api_key = api_key
        self.basic_auth = basic_auth
        self.headers = None
        self.is_authen = False

        if self.es_endpoint is not None and self.api_key is not None:
            self.authen(self.es_endpoint, api_key=self.api_key)

        elif self.es_endpoint is not None and self.basic_auth is not None:
            self.authen(self.es_endpoint, basic_auth=self.basic_auth)

        print(self.__repr__())
        if not self.is_authen:
            print(
                "Please authenticate using `authen()` method, or pass in the credentials at initialisation."
            )

    def authen(
        self,
        es_endpoint: str,
        api_key: str | None = None,
        basic_auth: tuple[str, str] | None = None,
    ):
        """Authenticates to the Elasticsearch endpoint

        Supports 2 types of authentication:
            - using an API key (`api_key`)
            - using a username and password (`basic_auth`)

        Parameters
        ----------
            - es_endpoint (str): Elasticsearch endpoint

            - api_key (str): API key for authentication (for API key authentication)

            - basic_auth (tuple[str, str]): A tuple of (username, password) (for basic authentication)

        Returns
        -------
            Response
        """

        if es_endpoint is None:
            raise ValueError("`es_endpoint` must have a value.")

        if api_key is None and basic_auth is None:
            raise ValueError("Either `api_key` or `basic_auth` must be passed in.")

        self.es_endpoint = es_endpoint
        self.api_key = api_key
        self.basic_auth = basic_auth

        if self.api_key:
            self.headers = {"Authorization": f"ApiKey {self.api_key}"}

        elif self.basic_auth:
            if not isinstance(self.basic_auth, tuple) or len(self.basic_auth) != 2:
                raise ValueError("`basic_auth` must be a tuple with a length of 2.")

            user_pass = ":".join(self.basic_auth)
            self.user_pass = base64.b64encode(user_pass.encode("utf-8")).decode()
            self.headers = {"Authorization": f"Basic {self.user_pass}"}

        auth_resp = requests.head(self.es_endpoint, headers=self.headers)

        if auth_resp.status_code == 200:
            self.is_authen = True
            return auth_resp
        else:
            raise Exception(
                f"Authentication failed: {auth_resp.status_code} - {auth_resp.text}"
            )

    def delete_index(self, index_name: str):
        """Deletes an index, if available

        Parameters
        ----------
            - index_name (str): Index name to delete

        Returns
        -------
        Response
        """

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, index_name)

        delete_response = requests.delete(es_url, headers=self.headers)
        if delete_response.status_code == 200:
            print(f"Index '{index_name}' index successfully deleted.")
            return delete_response
        else:
            raise Exception(
                f"Failed to delete index '{index_name}': {delete_response.status_code} - {delete_response.text}"
            )

    def create_index(
        self, index_name: str, json_mapping: dict, replace_if_exists: bool = False
    ):
        """Creates an index with a specified mapping

        Parameters
        ----------
            - index_name (str): Index name to create

            - json_mapping (dict): A dictionary mapping for the index

            - replace_if_exists (bool): Whether or not to delete and re-create a new index if already exists (Default = False)

        Returns
        -------
            Response
        """

        self._check_authen()

        create_headers = self.headers.copy()
        create_headers["Content-Type"] = "application/json"
        es_url = ppath.join(self.es_endpoint, index_name)

        if replace_if_exists:
            # Test whether it already exists
            test_response = requests.head(es_url, headers=self.headers)

            if test_response.status_code == 200:  # exists
                print(f"Found the existing index '{index_name}'")
                # Delete the existing index and create a new one
                _ = self.delete_index(index_name)

        # Create with the mapping
        create_response = requests.put(
            es_url, headers=create_headers, json=json_mapping
        )

        if create_response.status_code == 200:
            print(f"Index '{index_name}' has been created successfully.")

        return create_response

    def index_doc(self, index_name: str, doc_data: dict, _id: str | None = None):
        """Index one document to the specified index

        Parameters
        ----------
        index_name: str
            Index name to put/index data into

        doc_data: dict
            The data in JSON/dict format with the correct fields as the mapping

        _id: str | None (Default = None)
            The _id for the Elasticsearch index

            If None is provided, the `_id` of the document will be auto-generated.

        Returns
        -------
        Response
            The response of this index operation call
        """

        self._check_authen()

        index_headers = self.headers.copy()
        index_headers["Content-Type"] = "application/json"

        es_url = ppath.join(self.es_endpoint, index_name, "_doc")
        if _id:
            es_url += f"/{_id}"
            index_response = requests.put(es_url, headers=index_headers, json=doc_data)

        index_response = requests.post(es_url, headers=index_headers, json=doc_data)

        if index_response.status_code in (200, 201):
            print(f"The data has been successfully indexed into {index_name}")

        return index_response

    def update_doc(
        self,
        index_name: str,
        doc_data: dict,
        _id: str,
        detect_noop: bool = True,
        doc_as_upsert: bool = True,
    ):
        """Update one document to the specified index

        Parameters
        ----------
        index_name: str
            Index name to update data into

        doc_data: dict
            The data in JSON/dict format with the correct fields as the mapping

        _id: str
            The _id to update

            If None is provided, the `_id` of the document will be auto-generated.

        detect_noop: bool (Default = True)
            If `True`, the result in the response is set to noop (no operation) when there are no changes to the document

        doc_as_upsert: bool (Default = True)
            If True, use the contents of `doc_data` as the value of upsert operation

        Returns
        -------
        Response
            The response of this update operation call
        """

        self._check_authen()

        index_headers = self.headers.copy()
        index_headers["Content-Type"] = "application/json"

        es_url = ppath.join(self.es_endpoint, index_name, "_update", _id)

        json_body = {
            "detect_noop": detect_noop,
            "doc_as_upsert": doc_as_upsert,
            "doc": doc_data,
        }

        update_response = requests.post(es_url, headers=index_headers, json=json_body)

        if update_response.status_code == 200:
            print(f"The doc_id: {_id} has been updated at index: {index_name}")

        return update_response

    @staticmethod
    def generate_bulk_payload(
        index_name: str,
        data: list[dict],
        method: Literal["index", "update"] = "index",
        id_key: str | None = None,
    ):
        """Generates bulk payload data, for _bulk API"""

        # bulk_data = ""
        lines = []

        for item in data:
            # First row of the bulk
            if id_key is not None:
                _id = item[id_key]
                action = {method: {"_index": index_name, "_id": _id}}
            else:
                action = {method: {"_index": index_name}}

            # Second row (actual data) of the bulk
            if method == "update":
                # bulk_data += f"{json.dumps(action)}\n"
                bulk_data = json.dumps(
                    {"doc": item, "doc_as_upsert": True},
                    ensure_ascii=False,
                )
            else:
                # bulk_data = json.dumps(action)}\n{
                bulk_data = json.dumps(item, ensure_ascii=False)

            lines.append(json.dumps(action))
            lines.append(bulk_data)

        return "\n".join(lines) + "\n"

    def bulk_index(
        self,
        index_name: str,
        data: list[dict],
        id_key: str | None = None,
        routing_key: str | None = None,
        chunk_size: int = 10,
        refresh: Literal["true", "wait_for"] = "true",
    ):
        """Bulk index/update documents

        Parameters
        ----------
            - index_name (str): Index name to update to

            - data (list[dict]): A list of dictionary data to update to the index

            - id_key (str | None): Key of the dictionary data to use for documents' `_id` (Default = None)
                If None is provided, it will auto increment

            - routing_key (str | None): Routing name (shard) for all the data in this batch (Default = None)

            - chunk_size (int): The amount of data to be bulk updated in a single loop

            - refresh (Literal["true", "wait_for"]): Refresh parameter for Elasticsearch update/index API

        Returns
        -------
            Response
        """

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, index_name, "_bulk")
        es_url += f"?refresh={refresh}"

        if routing_key:
            es_url += f"&routing={routing_key}"

        i = 0
        while i < len(data):
            data_chunk = data[i : i + chunk_size]
            bulk_payload = self.generate_bulk_payload(
                index_name, data_chunk, method="index", id_key=id_key
            )

            headers = self.headers.copy()
            headers["Content-Type"] = "application/json"

            # Send bulk request
            response = requests.post(es_url, headers=headers, data=bulk_payload)

            if response.status_code != 200:
                print(
                    f"Error indexing batch {i} to {i + chunk_size - 1}: {response.text}"
                )
                return response
            else:
                print(f"Successfully indexed batch {i} to {i + chunk_size - 1}")

            i += chunk_size

        print("Finished updating all batches.")
        r = RequestResponse(
            status_code=200,
            content=json.dumps({"message": "All chunks are updated successfully."}),
        )
        return r

    def bulk_update(
        self,
        index_name: str,
        data: list[dict],
        id_key: str,
        routing_key: str | None = None,
        chunk_size: int = 100,
        refresh: Literal["true", "wait_for"] = "true",
    ):
        """Bulk index/update documents

        Parameters
        ----------
            - index_name (str): Index name to update to

            - data (list[dict]): A list of dictionary data to update to the index

            - id_key (str): Key of the dictionary data to use for documents' `_id`

            - routing_key (str | None): Routing name (shard) for all the data in this batch (Default = None)

            - chunk_size (int): The amount of data to be bulk updated in a single loop

            - refresh (Literal["true", "wait_for"]): Refresh parameter for Elasticsearch update/index API

        Returns
        -------
            Response
        """

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, index_name, "_bulk")
        es_url += f"?refresh={refresh}"

        if routing_key:
            es_url += f"&routing={routing_key}"

        i = 0
        while i < len(data):
            data_chunk = data[i : i + chunk_size]
            bulk_payload = self.generate_bulk_payload(
                index_name, data_chunk, method="update", id_key=id_key
            )

            headers = self.headers.copy()
            headers["Content-Type"] = "application/json"

            # Send bulk request
            response = requests.post(es_url, headers=headers, data=bulk_payload)

            if response.status_code != 200:
                print(
                    f"Error indexing batch {i} to {i + chunk_size - 1}: {response.text}"
                )
                return response
            else:
                print(f"Successfully indexed batch {i} to {i + chunk_size - 1}")

            i += chunk_size

        print("Finished updating all batches.")
        r = RequestResponse(
            status_code=200,
            content=json.dumps({"message": "All chunks are updated successfully."}),
        )
        return r

    def search(self, index_name: str, dsl: dict) -> SearchResults:
        """Search API to a speicified index and DSL query

        Parameters
        ----------
            - index_name (str): Index name to search from

            - dsl (dict): A DSL query

        Returns
        -------
            SearchResults object

        Example
        -------
            Example of using the results object retrieved from the `search()`

            ```
            dsl = {"query": {"match": ...}}
            results = els.search("my_index", dsl)

            sources = results.get_sources(include_id=True, include_score=True)
            df = results.to_dataframe(include_id=True, include_score=True)
            ```
        """

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, index_name, "_search")
        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"

        results = requests.post(url=es_url, headers=headers, data=json.dumps(dsl))
        search_results = SearchResults(results)
        return search_results

    def explain(
        self,
        index_name: str,
        doc_id: str,
        dsl: dict,
        routing: str | None = None,
    ) -> ExplainResult:
        """Explain API to a specified index and DSL query

        Parameters
        ----------
            - index_name (str): Index name to search from

            - doc_id (str): The document's `_id` to be explained by the DSL query

            - dsl (dict): A DSL query

            - routing (str | None): The routing (shard) name, if routing is required (Default = None)

        Returns
        -------
            ExplainResult object
        """

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, index_name, "_explain", doc_id)
        params = {"routing": routing} if routing else {}

        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"

        dsl = {"query": dsl["query"]}

        results = requests.get(
            es_url, params=params, headers=headers, data=json.dumps(dsl)
        )
        return ExplainResult(results)

    def count(self, index_name: str, dsl: dict | None = None) -> CountResults:
        """Returns the documents count in the specified `index_name`

        Parameters
        ----------
        index_name: str
            Index to count from

        dsl: dict | None (Default = None)
            DSL query you want to count the results of. If None, return the whole documents count.

        Returns
        -------
        CountResults
            CountResults object
        """

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, index_name, "_count")
        if not dsl or not isinstance(dsl, dict):
            results = requests.get(es_url, headers=self.headers)
            return CountResults(results)

        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"

        dsl = {"query": dsl["query"]}

        results = requests.post(es_url, headers=headers, data=json.dumps(dsl))
        return CountResults(results)

    def get_cat(self) -> CatResults:
        """Returns the stats of all the indices

        The `store_size` is in mb.

        Returns
        -------
            CatResults object
        """

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, "_cat/indices?format=json&bytes=mb")
        results = requests.get(es_url, headers=self.headers)

        return CatResults(results)

    def get_stats(self, index_name: str):
        """Returns the detailed stats of a specific index"""

        self._check_authen()

        es_url = ppath.join(self.es_endpoint, index_name, "_stats")
        results = requests.get(es_url, headers=self.headers)

        return StatsResults(results)

    def _check_authen(self):
        if not self.is_authen:
            raise Exception("Please authenticate first using `authen()` method.")

    def __repr__(self):
        return f"<ELS at {self.es_endpoint}, Authen: {self.is_authen}>"
