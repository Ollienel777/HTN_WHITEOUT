"""Every number the search policy has, in one place.

``hackathon/SPEC.md`` §5: policy parameters live in **one** dataclass,
serialisable to and from JSON, and nothing else configures the policy. The
tuner writes that JSON; a caller that wants a different operating point passes
a different :class:`SearchParams` and changes no code.

The defaults are argued, not guessed, and every argument is written down
beside the field it justifies. Where a number rests on something we measured
off the arena, the measurement is named.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["DEFAULT_SEARCH_PARAMS", "ParamsError", "SearchParams"]


class ParamsError(ValueError):
    """A parameter set that cannot describe a search."""


@dataclass(frozen=True)
class SearchParams:
    """The search policy's operating point.

    :param segment_m: how finely the channel is cut into segments, metres.
    :param stale_horizon_s: how long a segment takes to become fully
        uninteresting-to-interesting again.
    :param travel_cost_per_km: how much a kilometre of transit discounts a
        segment's score.
    :param hysteresis_margin: how much better a new target must be before an
        asset is re-tasked, as a fraction of its current target's score.
    :param tower_reach_m: how far a tower's camera is treated as useful.
    :param plane_min_reach_m: how far ahead the fixed-wing looks for work,
        so it is not handed a segment it is already on top of.
    :param search_alt_m: altitude for a searching aircraft, metres.
    :param hold_alt_m: altitude for the quadcopter holding a contact.
    :param look_radius_m: how close an asset must be to a segment for that
        segment to count as looked at this tick.
    """

    segment_m: float = 250.0
    stale_horizon_s: float = 240.0
    travel_cost_per_km: float = 0.08
    hysteresis_margin: float = 0.25
    tower_reach_m: float = 4000.0
    plane_min_reach_m: float = 600.0
    search_alt_m: float = 120.0
    hold_alt_m: float = 60.0
    look_radius_m: float = 450.0

    def __post_init__(self) -> None:
        positive = (
            ("segment_m", self.segment_m),
            ("stale_horizon_s", self.stale_horizon_s),
            ("tower_reach_m", self.tower_reach_m),
            ("search_alt_m", self.search_alt_m),
            ("hold_alt_m", self.hold_alt_m),
            ("look_radius_m", self.look_radius_m),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ParamsError(f"{name} must be finite and positive, got {value!r}")
        non_negative = (
            ("travel_cost_per_km", self.travel_cost_per_km),
            ("hysteresis_margin", self.hysteresis_margin),
            ("plane_min_reach_m", self.plane_min_reach_m),
        )
        for name, value in non_negative:
            if not math.isfinite(value) or value < 0.0:
                raise ParamsError(f"{name} must be finite and non-negative, got {value!r}")

    def to_json(self) -> str:
        """Serialise. What the tuner writes and the runner reads."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SearchParams:
        """Parse, rejecting unknown keys rather than ignoring them.

        A parameter file written against a later version of this dataclass
        would otherwise load with the unknown field silently dropped, and the
        run would be reported as using an operating point it was not using.
        """
        try:
            raw: Any = json.loads(text)
        except json.JSONDecodeError as error:
            raise ParamsError(f"parameters are not JSON: {error}") from error
        if not isinstance(raw, dict):
            raise ParamsError(f"parameters must be a JSON object, got {type(raw).__name__}")
        known = {field for field in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ParamsError(
                f"unknown parameter(s): {', '.join(unknown)}. Dropping them silently would "
                f"report a run as using an operating point it was not using"
            )
        return cls(**raw)


#: The operating point in use. Named so that a caller reads
#: ``DEFAULT_SEARCH_PARAMS`` and knows it is a default rather than a constant.
DEFAULT_SEARCH_PARAMS = SearchParams()
