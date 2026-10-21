"""logos - logical reasoning + fact-checking SDK for the lair.

Layered so that what CAN be decided deterministically is, and what cannot be is
declared rather than faked.

    from logos import capabilities, syllogism
    capabilities("logic.syllogism")   # what it proves, and what it cannot
    syllogism.explain("AAA", 1)       # Barbara, valid
"""

from __future__ import annotations

from .honesty import CAPABILITIES, can_prove, capabilities
from .logic import syllogism

__version__ = "0.1.0"

__all__ = ["capabilities", "CAPABILITIES", "can_prove", "syllogism", "__version__"]
