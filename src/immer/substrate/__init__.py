"""Substrate: the physics of a living system, none of its politics.

Design rules (agreed 2026-08-21):

- The substrate owns *contracts and continuity*, never behaviour policy.
- No turn loop, no speech censor, no memory schema. What the organism
  attends to, when it consolidates, which organs it mounts — that is its
  first acquired competence, not our architecture.
- Chat is one input channel among several, not the purpose of the system.
"""

from .bus import EventBus, Event, Priority
from .daemon import LifeDaemon, LifeStatePort, OrganRack

__all__ = ["EventBus", "Event", "Priority", "LifeDaemon", "LifeStatePort", "OrganRack"]
