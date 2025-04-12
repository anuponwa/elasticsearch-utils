from typing import Literal, TypedDict


class DetailsDict(TypedDict):
    value: float
    description: str
    details: list["DetailsDict"]


class FieldScoreDict(TypedDict):
    field: str
    clause: str
    type: Literal[r"value", "boost", "idf", "tf"]
    value: int | float


class ScoreSummaryDict(TypedDict):
    value: float
    boost: float
