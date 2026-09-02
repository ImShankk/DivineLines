"""Soccer Match Center: the observable record of a match, assembled once.

The pieces here answer a question each, and nothing is drawn that the data
cannot support:

* :mod:`spatial`  — where did events happen? (real coordinates, one transform)
* :mod:`momentum` — how did control of the match change over time?
* :mod:`stats`    — what did the box score say, per team and per player?
* :mod:`quality`  — what do we actually have for this match, and what is missing?
* :mod:`service`  — assemble all of it for one fixture, respecting chronology.
* :mod:`report`   — the same material as prose a person can read.

Two rules hold across the whole package:

1. A metric with no source is not rendered. It reports ``NO_DATA`` with the
   reason, which is more useful than an empty chart and far more honest than
   a plausible-looking invention.
2. Every read takes an ``as_of`` and a match-clock bound. Replay is enforced
   here, on the server, not by hiding elements in React.
"""

from .momentum import MOMENTUM_VERSION, momentum_series
from .quality import match_intelligence
from .service import match_center, match_events, match_momentum, match_passes
from .spatial import PITCH_LENGTH, PITCH_WIDTH, normalise_point, shot_map

__all__ = [
    "MOMENTUM_VERSION",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "match_center",
    "match_events",
    "match_intelligence",
    "match_momentum",
    "match_passes",
    "momentum_series",
    "normalise_point",
    "shot_map",
]
