"""The coordinator's decisions: where every asset goes next.

``hackathon/SPEC.md`` §4. Allocation, routing and re-tasking live here, and
nothing here imports from ``whiteout.sim`` (§5) or knows how belief is stored.
"""

from whiteout.policy.params import DEFAULT_SEARCH_PARAMS, ParamsError, SearchParams
from whiteout.policy.search import AssetRole, SearchPolicy, Segment

__all__ = [
    "DEFAULT_SEARCH_PARAMS",
    "AssetRole",
    "ParamsError",
    "SearchParams",
    "SearchPolicy",
    "Segment",
]
