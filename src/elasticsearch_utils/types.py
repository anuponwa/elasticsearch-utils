from typing import TypedDict, NotRequired


class ExplainDetailsDict(TypedDict):
    value: float
    description: str
    details: list["ExplainDetailsDict"]


# class FieldScoreDict(TypedDict):
#     field: str
#     clause: str
#     type: Literal[r"value", "boost", "idf", "tf"]
#     value: int | float


# class ScoreSummaryDict(TypedDict):
#     value: float
#     boost: float


# Results from _cat
class CatDict(TypedDict):
    health: str
    status: str
    index: str
    uuid: str
    pri: str
    rep: str
    docs_count: str
    docs_deleted: str
    store_size: str
    pri_store_size: str


# Results from _stats
class DocsStats(TypedDict):
    count: int
    deleted: int


class StoreStats(TypedDict):
    size_in_bytes: int
    total_data_set_size_in_bytes: int
    reserved_in_bytes: int


class IndexingStats(TypedDict):
    index_total: int
    index_time_in_millis: int
    index_current: int
    index_failed: int
    delete_total: int
    delete_time_in_millis: int
    delete_current: int
    noop_update_total: int
    is_throttled: bool
    throttle_time_in_millis: int


class GetStats(TypedDict):
    total: int
    time_in_millis: int
    exists_total: int
    exists_time_in_millis: int
    missing_total: int
    missing_time_in_millis: int
    current: int


class SearchStats(TypedDict):
    open_contexts: int
    query_total: int
    query_time_in_millis: int
    query_current: int
    fetch_total: int
    fetch_time_in_millis: int
    fetch_current: int
    scroll_total: int
    scroll_time_in_millis: int
    scroll_current: int
    suggest_total: int
    suggest_time_in_millis: int
    suggest_current: int


class MergesStats(TypedDict):
    current: int
    current_docs: int
    current_size_in_bytes: int
    total: int
    total_time_in_millis: int
    total_docs: int
    total_size_in_bytes: int
    total_stopped_time_in_millis: int
    total_throttled_time_in_millis: int
    total_auto_throttle_in_bytes: int


class RefreshStats(TypedDict):
    total: int
    total_time_in_millis: int
    external_total: int
    external_total_time_in_millis: int
    listeners: int


class FlushStats(TypedDict):
    total: int
    periodic: int
    total_time_in_millis: int


class WarmerStats(TypedDict):
    current: int
    total: int
    total_time_in_millis: int


class QueryCacheStats(TypedDict):
    memory_size_in_bytes: int
    total_count: int
    hit_count: int
    miss_count: int
    cache_size: int
    cache_count: int
    evictions: int


class FieldDataStats(TypedDict):
    memory_size_in_bytes: int
    evictions: int


class SegmentsStats(TypedDict):
    count: int
    memory_in_bytes: int
    terms_memory_in_bytes: int
    stored_fields_memory_in_bytes: int
    term_vectors_memory_in_bytes: int
    norms_memory_in_bytes: int
    points_memory_in_bytes: int
    doc_values_memory_in_bytes: int
    index_writer_memory_in_bytes: int
    version_map_memory_in_bytes: int
    fixed_bit_set_memory_in_bytes: int
    max_unsafe_auto_id_timestamp: int
    file_sizes: dict[str, int]


class TranslogStats(TypedDict):
    operations: int
    size_in_bytes: int
    uncommitted_operations: int
    uncommitted_size_in_bytes: int
    earliest_last_modified_age: int


class RequestCacheStats(TypedDict):
    memory_size_in_bytes: int
    evictions: int
    hit_count: int
    miss_count: int


class RecoveryStats(TypedDict):
    current_as_source: int
    current_as_target: int
    throttle_time_in_millis: int


class ShardsMetadata(TypedDict):
    total: int
    successful: int
    failed: int


class ShardStats(TypedDict):
    total_count: int


class StatsBlock(TypedDict):
    docs: DocsStats
    shard_stats: ShardStats
    store: StoreStats
    indexing: IndexingStats
    get: GetStats
    search: SearchStats
    merges: MergesStats
    refresh: RefreshStats
    flush: FlushStats
    warmer: WarmerStats
    query_cache: QueryCacheStats
    fielddata: FieldDataStats
    completion: dict[str, int]
    segments: SegmentsStats
    translog: TranslogStats
    request_cache: RequestCacheStats
    recovery: RecoveryStats


class IndexStats(TypedDict):
    uuid: NotRequired[str]
    primaries: StatsBlock
    total: StatsBlock


class ShardsStats(TypedDict):
    total: int
    successful: int
    failed: int


class StatsResultsDict(TypedDict):
    _shards: ShardsMetadata
    _all: IndexStats
    indices: dict[str, IndexStats]


# count results
class CountShardsDict(TypedDict):
    total: int
    successful: int
    skipped: int
    failed: int


class CountDict(TypedDict):
    count: int
    _shards: CountShardsDict
