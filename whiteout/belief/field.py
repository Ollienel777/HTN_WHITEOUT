"""The seam between the belief field and everything that reads it.

The policy must not know what shape the field is. Today it is a ribbon over
the water of Bellot Strait (:mod:`whiteout.belief.grid`); if the arena turns
out to need a second basin, or a particle filter, or a coarser grid, the only
thing that may have to change is the class behind this protocol.

So the protocol is written in the terms the *callers* have — **lat/lon and
metres**, which is what MAVLink, ``whiteout/vision/`` and the tracks API all
speak (``ARENA.md`` §5) — and never in cells, indices or axes.

:meth:`BeliefField.update_likelihood` is the general one, and it is the reason
this protocol is not just three accessors. Issue #13's negative-information
update has to multiply the field by "how likely is it that a camera pointed
*there* saw nothing?", cell by cell, and it must be able to do that without
knowing that cells exist. A callable from a position to a likelihood is that,
exactly, and :meth:`update_detection` is implemented in terms of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["BeliefError", "BeliefField", "BeliefPeak", "Likelihood"]


class BeliefError(ValueError):
    """An update would leave the field with no probability mass, or is ill-posed.

    Raised rather than absorbed: a field that has been multiplied by an
    all-zero likelihood cannot be renormalised, and the alternatives — an
    all-zero field, or silently resetting to uniform — both make the fleet
    search confidently wrong water. The caller's response is to discard the
    evidence, which it can only do if it hears about it.
    """


#: A likelihood over positions: ``(lat_deg, lon_deg) -> non-negative weight``.
#: Only ratios matter, so it need not integrate to anything.
Likelihood = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class BeliefPeak:
    """The most probable place the vessel is, and how much belief sits there.

    ``probability`` is the *mass* of the single most probable region, not a
    density, so it is a number in ``[0, 1]`` a threshold can be written
    against. It does depend on how finely the field is resolved — a finer
    field divides the same belief into smaller pieces — so two fields of
    different resolution are compared by :meth:`BeliefField.entropy`, not by
    this.
    """

    lat_deg: float
    lon_deg: float
    probability: float


@runtime_checkable
class BeliefField(Protocol):
    """What the policy is allowed to know about the belief field."""

    def mass(self) -> float:
        """Total probability mass. 1.0 within floating-point tolerance, always."""

    def entropy(self) -> float:
        """Shannon entropy of the field, nats. Maximal when belief is uniform."""

    def peak(self) -> BeliefPeak:
        """The most probable position, and its share of the mass."""

    def probability_at(self, lat_deg: float, lon_deg: float) -> float:
        """Mass of the region containing this position; 0.0 on land or off the arena."""

    def diffuse(self, dt_s: float) -> None:
        """Advance the field by ``dt_s`` seconds of the vessel's random walk."""

    def update_detection(self, lat_deg: float, lon_deg: float, *, sigma_m: float) -> None:
        """Fold in a positive detection at a position, with its one-sigma error."""

    def update_likelihood(self, likelihood: Likelihood) -> None:
        """Multiply the field by an arbitrary likelihood and renormalise."""
