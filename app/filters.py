# app/filters.py
from __future__ import annotations
from typing import Any, Dict, List, Union

Value = Union[str, int, float, bool]
MaybeList = Union[Value, List[Value]]

# Map CLI keys → KB metadata fields (adjust to your actual field names)
META_MAP = {
    "year": "year",
    "topic": "topic",
    "source": "source",
    "part": "part",
}


def build_vector_filter(**kwargs: MaybeList) -> Dict[str, Any] | None:
    # No metadata in KB => no server-side filter
    return None

# def _to_list(v: MaybeList) -> List[Value]:
#     if v is None:
#         return []
#     return v if isinstance(v, list) else [v]
#
# def _coerce_value(key: str, v: Value) -> Value:
#     # Cast year to int if possible; keep others as-is
#     if key == "year":
#         try:
#             return int(v)
#         except Exception:
#             return str(v)
#     # normalize part like "b" -> "B" (optional)
#     if key == "part" and isinstance(v, str):
#         return v.strip().upper()
#     if isinstance(v, str):
#         return v.strip()
#     return v
#
# def build_vector_filter(**kwargs: MaybeList) -> Dict[str, Any] | None:
#     """
#     Build Bedrock KB RetrievalFilter.
#     - Single value -> {"equals": {"key": field, "value": raw_json_value}}
#     - Multi value  -> {"in":     {"key": field, "value": [raw_json_values...]}}
#     - Combine across fields with {"andAll": [ ... ]}
#     """
#     clauses: List[Dict[str, Any]] = []
#
#     for cli_key, raw in kwargs.items():
#         if raw is None:
#             continue
#         field = META_MAP.get(cli_key)
#         if not field:
#             continue
#
#         vals = [x for x in _to_list(raw) if x is not None and str(x) != ""]
#         if not vals:
#             continue
#
#         # Coerce values (e.g., year -> int)
#         coerced = [_coerce_value(cli_key, v) for v in vals]
#
#         if len(coerced) == 1:
#             clauses.append({"equals": {"key": field, "value": coerced[0]}})
#         else:
#             clauses.append({"in": {"key": field, "value": coerced}})
#
#     if not clauses:
#         return None
#     if len(clauses) == 1:
#         return clauses[0]
#     return {"andAll": clauses}
