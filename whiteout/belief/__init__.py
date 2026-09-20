"""Belief: where the vessel probably is.

:class:`~whiteout.belief.field.BeliefField` is the seam the policy sees.
:class:`~whiteout.belief.grid.ChannelBeliefGrid` is today's implementation, a
ribbon over the water of :data:`~whiteout.belief.geometry.DEFAULT_STRAIT`.
"""

from whiteout.belief.encode import belief_frame, belief_geometry
from whiteout.belief.field import BeliefError, BeliefField, BeliefPeak, Likelihood
from whiteout.belief.geometry import (
    DEFAULT_STRAIT,
    ChannelPoint,
    ChannelVertex,
    GeometryError,
    StraitGeometry,
)
from whiteout.belief.grid import ChannelBeliefGrid

__all__ = [
    "DEFAULT_STRAIT",
    "BeliefError",
    "BeliefField",
    "BeliefPeak",
    "ChannelBeliefGrid",
    "ChannelPoint",
    "ChannelVertex",
    "GeometryError",
    "Likelihood",
    "StraitGeometry",
    "belief_frame",
    "belief_geometry",
]
