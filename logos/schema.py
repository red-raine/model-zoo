"""logos.schema — the typed contract every verification component speaks.

WHY A SCHEMA AND NOT JUST DICTS
-------------------------------
Every failure in this project so far has been a **units-and-identity** failure, not an
algorithm failure:

  * a generation parser that matched any label found in degenerate text (an identity failure)
  * fp16 weights meeting fp32 class weights (a dtype units failure)
  * a checkpoint whose label order was `[1,0,2,3]` (an identity failure that read as 0.106)
  * a calibration split taken from the corpus the transferred model was trained on (an
    identity failure: "calib" did not mean what the code assumed)
  * a correctness head whose features were all functions of the same forward pass (an
    information failure)

A pydantic schema does not fix reasoning, but it makes the third and fourth impossible:
`Split.CALIB` and `Split.TRANSFER_CONTAMINATED` are different types, not two strings. And it
gives the audit layer a stable object to serialise — the 2026 survey of 121 systems
(arXiv:2609.23239) found that a "warrant" (what was guaranteed, on what grounds) is the
artefact almost every pipeline fails to emit.

DESIGN RULE
-----------
Nothing here computes a verdict. This module is pure data with validation, so it can be
imported by the symbolic engine, the neural head, the aggregator and the audit log without
creating a cycle or smuggling in a model call.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- enums
Label = Literal["Supported", "Refuted", "Not Enough Evidence", "Conflicting"]
LABELS: tuple[Label, ...] = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting")
L2I = {lab: i for i, lab in enumerate(LABELS)}


class Split(str, Enum):
    """Which corpus slice a number came from.

    TRANSFER_CONTAMINATED exists as a first-class value because a transferred checkpoint may
    already have trained on what we call "train". Measured: CTU AVeriTeC-large scored 0.9260
    on our 500-row train carve and 0.8080 on dev. Calibrating on the former made the
    selective-risk controller promise 15% error and deliver 18.6%. The enum makes that
    mistake require typing its own name.
    """

    TRAIN_FIT = "train_fit"
    CALIB = "calib"
    DEV = "dev"
    TRANSFER_CONTAMINATED = "transfer_contaminated"


class EvidenceLabel(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    INSUFFICIENT = "insufficient"


class ChannelStatus(str, Enum):
    SETTLED = "settled"        # a deterministic channel proved it
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


# ------------------------------------------------------------------ deterministic layer
class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.end < self.start:
            raise ValueError(f"span end {self.end} < start {self.start}")
        return self


class Quantity(BaseModel):
    """A normalised numeric quantity.

    `raw` is kept for the audit trail; `value` is what arithmetic uses. "1.2 million" and
    "1,200,000" MUST produce the same `value` and `unit` -- that equality is the whole reason
    this class exists, and it is asserted in the self-test.
    """

    value: float
    unit: str = ""
    raw: str = ""
    span: Span | None = None

    @field_validator("unit")
    @classmethod
    def _norm_unit(cls, v: str) -> str:
        return v.strip().lower().replace("kilometres", "km").replace("kilometers", "km")

    def __eq__(self, other: object) -> bool:      # noqa: D105 - deliberate value equality
        if not isinstance(other, Quantity):
            return NotImplemented
        return abs(self.value - other.value) < 1e-9 and self.unit == other.unit

    def __hash__(self) -> int:
        return hash((round(self.value, 9), self.unit))


class TimeInterval(BaseModel):
    """ISO interval, half-open [start, end). `precision` guards the classic trap of treating
    'the early 1990s' as a point date."""

    start: str
    end: str
    precision: Literal["day", "month", "year", "decade", "century", "unknown"] = "unknown"
    raw: str = ""

    @model_validator(mode="after")
    def _ordered(self):
        if self.end < self.start:
            raise ValueError(f"interval end {self.end} < start {self.start}")
        return self


class EntityMention(BaseModel):
    text: str
    span: Span
    label: str = "ENT"
    wikidata_id: str | None = None

    @property
    def grounded(self) -> bool:
        return self.wikidata_id is not None


class NegationScope(BaseModel):
    """Token span the negation actually governs.

    Getting this wrong is the single most common silent error in rule-based fact checking:
    "the first man on the moon was not American" negates the predicate, not the entity.
    """

    span: Span
    cue: str


class ClaimIR(BaseModel):
    """Deterministic intermediate representation of a claim. No model, no GPU."""

    claim_id: str
    text: str
    entities: list[EntityMention] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    dates: list[TimeInterval] = Field(default_factory=list)
    negations: list[NegationScope] = Field(default_factory=list)
    modality: Literal["asserted", "hedged", "reported", "unknown"] = "unknown"
    comparator: Literal["gt", "lt", "eq", "ratio", "none"] = "none"

    @property
    def grounded_entities(self) -> int:
        return sum(1 for e in self.entities if e.grounded)


# ------------------------------------------------------------------------ neural layer
class EvidenceVerdict(BaseModel):
    """A SPAN-LEVEL verdict: what one retrieved snippet says about the claim.

    This is the layer the whole project was missing. The claim-level model produced one
    4-way label per claim and had to average its own uncertainty across ~4 snippets, which is
    why `Conflicting` (F1 .362) collapses: conflict is an AGGREGATION property, invisible to
    any single forward pass over concatenated text.
    """

    evidence_id: str
    text: str
    label: EvidenceLabel
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = ""
    source_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: str | None = None
    split: Split | None = None

    @property
    def weight(self) -> float:
        """Confidence x reliability. Deliberately not a learned function."""
        return self.confidence * self.source_reliability


class ChannelResult(BaseModel):
    """One deterministic channel's ruling on one claim. `warrant` is the audit artefact."""

    channel: str
    status: ChannelStatus
    detail: str = ""
    warrant: dict = Field(default_factory=dict)
    implies: Label | None = None


# ------------------------------------------------------------------------ belief layer
class Belief(BaseModel):
    """The four NAMED quantities that "this looks fishy" silently conflates.

    I built a 15-feature logistic head over this concept three times and it lost to free
    max-probability three times. The reason is not that the idea is wrong -- it is that
    "fishiness" is four different questions with four different engines, and a linear head
    on correlated summary features cannot separate them:

      p_verdict_wrong   the discriminative question   -> calibrated encoder (max_p)
      p_prior_mismatch  "does this claim look made up" -> claim-only model + surface probes
      p_evidence_gap    "did retrieval fail"          -> retrieval coverage / sufficiency
      p_epistemic       "is the encoder out of depth" -> seed-ensemble disagreement

    Each has its own validation target, so none can hide inside the others' average.
    """

    p_verdict_wrong: float = Field(ge=0.0, le=1.0)
    p_prior_mismatch: float | None = Field(default=None, ge=0.0, le=1.0)
    p_evidence_gap: float | None = Field(default=None, ge=0.0, le=1.0)
    p_epistemic: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def fishy(self) -> float:
        """A conjunction-flavoured combination, NOT an average.

        Averaging lets one confident component hide another screaming one: a claim with
        p_evidence_gap=0.95 and p_prior_mismatch=0.9 would average to ~0.9 with three easy
        components and look acceptable. A noisy-OR says "any one of these being likely is
        enough to be suspicious", which is the behaviour we actually want from a gate.
        """
        parts = [v for v in (self.p_prior_mismatch, self.p_evidence_gap,
                             self.p_epistemic) if v is not None]
        if not parts:
            return 0.0
        keep = 1.0
        for p in parts:
            keep *= 1.0 - p
        return 1.0 - keep


# ----------------------------------------------------------------------------- warrant
class Warrant(BaseModel):
    """Everything needed to audit a verdict after the fact.

    Emitted unconditionally, including for abstentions. `flip_condition` answers the question
    an auditor actually asks: what would have had to be different?
    """

    claim_id: str
    verdict: Label | None
    rule: str = ""
    fired_channels: list[str] = Field(default_factory=list)
    used_evidence: list[str] = Field(default_factory=list)
    belief: Belief | None = None
    flip_condition: str = ""
    split: Split = Split.DEV
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _abstention_needs_a_reason(self):
        if self.verdict is None and not self.notes:
            raise ValueError("an abstention must record why it abstained")
        return self


if __name__ == "__main__":  # tiny smoke test
    q1 = Quantity(value=1.2e6, unit="people", raw="1.2 million")
    q2 = Quantity(value=1200000.0, unit="people", raw="1,200,000")
    assert q1 == q2, "quantity normalisation is broken"
    b = Belief(p_verdict_wrong=0.4, p_prior_mismatch=0.9, p_evidence_gap=0.95)
    assert b.fishy > 0.99, b.fishy
    print("schema OK", b.fishy)
