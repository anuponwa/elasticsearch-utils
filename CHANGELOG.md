# Change log
## 0.5.1
* Provide more attributes in `ExplainResults`
* Add docs to some functions

## 0.5.0
* Re-design the `ExplainResults` object methods
* Support the more correct and insightful flatten, details and summary of the scores

## 0.4.2
* Fix the main import `ESClient`
* Add count with DSL functionality

## 0.4.1
* Fully change to elasticsearch-utils and update doc

## 0.4.0
* Change to ES convention, the main class has been changed to `ESClient`.

## 0.3.9
* Add `update_doc()` and `index_doc()` methods to update/index a single document

## 0.3.8
* Add `count()` method

## 0.3.7
* Add `get_stats()` `get_cat()` methods to get stats of the indices in the Elasticsearch server

## 0.3.6
* Fix generate bulk payload function

## 0.3.5
* Fix bulk payload for index method

## 0.3.4
* Fix bulk update/index methods

## 0.3.3
* Add bulk index

## 0.3.2
* Add support for default reindexing with `_id=None`

## 0.3.1
* Fix return json content error

## 0.3.0
* Change `replace_if_exists` to `False` by default when creating an index

## 0.2.0
* Add explain API
* Add support for explanation results in `SearchResults`
* Add functionalities to ``ExplainResult`
* Re-design how `SearchResult` returns `sources`

## 0.1.0
* First development version