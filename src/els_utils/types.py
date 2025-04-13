from typing import Literal, TypedDict


class ExplainDetailsDict(TypedDict):
    value: float
    description: str
    details: list["ExplainDetailsDict"]


class FieldScoreDict(TypedDict):
    field: str
    clause: str
    type: Literal[r"value", "boost", "idf", "tf"]
    value: int | float


class ScoreSummaryDict(TypedDict):
    value: float
    boost: float
