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
exactly, and :meth:`update_detection` is the same likelihood specialised: it
builds the identical weights vectorised and hands them to the same
multiply-and-renormalise step. It does **not** call
:meth:`update_likelihood`, so the per-cell finite/non-negative validation on
that path does not guard detections;
``test_update_likelihood_is_the_general_seam`` is what holds the two to the
same answer.
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
    field divides the same belief into smaller pieces — so a threshold written
    against it is a threshold written against one resolution.

    **:meth:`BeliefField.entropy` is not the way around that**, and an earlier
    version of this docstring said it was. Discrete Shannon entropy carries a
    ``log N`` term, so it moves with resolution at least as much. Measured on
    ``DEFAULT_STRAIT`` — one detection at ``sigma_m`` = 150 m, placed on the
    **water cell centre nearest mid-strait** at each resolution, starting from
    the uniform prior — it reads **0.703 nats at 400 m cells, 3.837 at 100 m
    and 5.183 at 50 m**.

    The position has to be stated because at 400 m the figure is not stable
    under it: the same detection swept over all 255 water-cell centres reads
    anywhere in **0.443 to 0.703 nats**, since a 150 m error inside a 400 m
    cell barely distinguishes one cell from its neighbours and the update is
    then a function of where in the cell the fix landed. Read the 400 m row as
    that range. ``tests/test_belief_grid.py`` re-measures all of it and checks
    this paragraph still says it.

    Both numbers on this protocol are resolution-dependent, there is no
    resolution-independent comparator here yet, and a policy that hard-codes a
    threshold against either is coupled to the field's internals through the
    back door. Normalising entropy by ``log(water_cells)``, or reporting a
    credible interval in metres, would be one; #13 and #14 should ask for it
    rather than assume it.
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
