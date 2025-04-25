# elasticsearch-utils

This library provides utilities for interacting with Elasticsearch and results retrived from it.

The core features are:
- Index creation (even if already exists)
- Index deletion
- Search (can include `explain: True` in the DSL query)
- Explain

## Installation

```
uv add elasticsearch-utils
```

## Example usage

### Instantiate
```python
from elasticsearch_utils import ESClient


# Using API key to authenticate
es = ESClient(ES_ENDPOINT, api_key=ES_API_KEY)

# Using basic authen with username and password
es = ELS(ES_ENDPOINT, basic_authen=(USERNAME, PASSWORD))
```
### Create index
```python
mapping = {"mappings": {"properties": {...}}}

es.create_index(index_name="my-index", json_mapping=mapping, replace_if_exists=True)
```

### Bulk update
```python
data = [{"some-id": "1", "field1": "some-value"}, {"some-id": "2", "field1": "another-value"}]

es.bulk_update(index_name="my-index", data=data, id_key="some-id")

# Routing option is also available if needed
es.bulk_update(index_name="my-index", data=data, id_key="some-id", routing_key="1")
```

### Results

```python
dsl = {"query": {...}}
results = es.search(INDEX_NAME, dsl)

# results: <SearchResults total_hits=510>

# Get the JSON
results.json

# Get the hits
result.hits

# Get the sources
results.get_sources(as_list=False)  # as_list can be `True` if you wish to get just a list of sources

# Get results in a DataFrame format
results.to_dataframe()

# You can also pass in just the columns you want
results.to_dataframe(columns=["field1"])

# If explain: True was passed in the DSL, you can also get explanations
results.get_explanations()  # This returns a dict of {`_id`: `<ExplainResult>`}
```

### Explanation

#### Using the `explain` API
```python
dsl = {"query": {...}}
explain = es.explain(INDEX_NAME, doc_id="75720", dsl=dsl, routing=None)

# Get the JSON
explain.json

# Get score
explain.score

# Get the explanation dict
explain.explanation

# Get the scores terms, e.g., field, term, score, boost, etc.
explain.get_field_details()

# Get the contributions summary for each field
explain.get_field_summary()
```

#### Using the explanation from the `search` API
```python
dsl = {"explain": True, "query": {...}}
results = es.search(INDEX_NAME, dsl=dsl)

# This returns a dict of {`_id`: `<ExplainResult>`}
explanations = results.get_explanations()

# You can do the same thing as the `ExplainResult` for each item
explanations["75720"].get_field_summary()  # Given that the key "75720" exists in the results
```