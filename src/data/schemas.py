"""Shared data shapes for the relation lie-detector pipeline, one per stage output.

Record files (a JSON array, one record per line):

    CleanRelation     stages 1-2  cleaned/clean_data.json, enriched/enriched_data.json
    Query             stage 3     queries/queries_{split}.json
    TrainRecord       stage 4     answers/{split}_records.json; a Query after the VLM has answered it
    ValidationAnswer  stages 7-8  validation/*.json; a re-asked or steered answer, cached

Result files (one JSON object):

    ValidationSettings  stage 7  validation/results/settings.json; the setting chosen for each method
    TestResults         stage 8  validation/results/test.json; metrics for the baseline, methods and controls

Hidden states are stored separately as a float16 array of shape [N, n_saved_layers, d_model]
(answers/{split}_hidden.npy), where row i belongs to line i of the TrainRecord file; layers.json lists the layers.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Predicate(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    FRONT = "FRONT"
    BACK = "BACK"
    ON = "ON"
    UNDER = "UNDER"

    @property
    def opposite(self) -> "Predicate":
        return _OPPOSITE[self]

    @property
    def phrase(self) -> str:
        """Wording used in a question: "Is the cup {phrase} the plate?"."""
        return _PHRASE[self]


_OPPOSITE = {
    Predicate.LEFT: Predicate.RIGHT,
    Predicate.RIGHT: Predicate.LEFT,
    Predicate.UP: Predicate.DOWN,
    Predicate.DOWN: Predicate.UP,
    Predicate.FRONT: Predicate.BACK,
    Predicate.BACK: Predicate.FRONT,
    Predicate.ON: Predicate.UNDER,
    Predicate.UNDER: Predicate.ON,
}

_PHRASE = {
    Predicate.LEFT: "to the left of",
    Predicate.RIGHT: "to the right of",
    Predicate.UP: "above",
    Predicate.DOWN: "below",
    Predicate.FRONT: "in front of",
    Predicate.BACK: "behind",
    Predicate.ON: "on",
    Predicate.UNDER: "under",
}

# Raw Visual Genome wordings accepted for each label, matched exactly after lowercasing and
# dropping "is/are/a/an/the". Each one is a canonical line in relationship_alias.txt.
# Labels are assigned from this map, not from the alias file, which merges
# "on top of" into "above" and "below" into "under".
_RAW_PHRASES = {
    Predicate.LEFT: ["left of", "to left of", "on left of", "on left side of"],
    Predicate.RIGHT: ["right of", "to right of", "on right of", "on right side of"],
    Predicate.UP: ["above", "over"],
    Predicate.DOWN: ["below"],
    Predicate.FRONT: ["in front of", "front of"],
    Predicate.BACK: ["behind"],
    Predicate.ON: ["on", "on top of"],
    Predicate.UNDER: ["under", "underneath", "beneath"],
}

RAW_TO_PREDICATE = {phrase: label for label, phrases in _RAW_PHRASES.items() for phrase in phrases}


class Source(str, Enum):
    ANNOTATION = "annotation"  # taken from a Visual Genome relationship
    BBOX = "bbox"  # derived from the positions of two object boxes


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Variant(str, Enum):
    TRUE = "true"  # the relation as it is; answer yes
    OPPOSITE = "opposite"  # predicate flipped; answer no
    SWAPPED = "swapped"  # subject and object swapped; answer no
    SWAPPED_OPPOSITE = "swapped_opposite"  # both swapped and flipped, so true again; answer yes

    @property
    def answer(self) -> "Answer":
        return Answer.YES if self in (Variant.TRUE, Variant.SWAPPED_OPPOSITE) else Answer.NO


class Answer(str, Enum):
    YES = "yes"
    NO = "no"


class _Record(BaseModel):
    # protected_namespaces=() lets older pydantic 2 accept the field name model_answer without a warning
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class BBox(_Record):
    """Box in pixels, as Visual Genome stores it: top-left corner plus size."""

    x: int
    y: int
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


class CleanRelation(_Record):
    rel_id: str  # Visual Genome relationship_id, or "{image_id}_{subj_id}_{obj_id}" for bbox rows
    image_id: int
    subj_id: int
    subj: str
    subj_bbox: BBox
    obj_id: int
    obj: str
    obj_bbox: BBox
    predicate_raw: str | None  # original Visual Genome wording; None for bbox rows
    label: Predicate
    source: Source
    split: Split

    @model_validator(mode="after")
    def _raw_matches_source(self):
        if (self.source is Source.ANNOTATION) != (self.predicate_raw is not None):
            raise ValueError("predicate_raw must be set for annotation rows and None for bbox rows")
        return self


class Query(_Record):
    query_id: str  # "{rel_id}_{variant}_{paraphrase}"
    rel_id: str
    image_id: int
    split: Split
    # the relation's two objects, copied so the image can be drawn without the clean data;
    # subj is boxed in red and obj in blue, whichever one the question names first
    subj: str
    subj_bbox: BBox
    obj: str
    obj_bbox: BBox
    label: Predicate  # the true relation, whichever variant is asked
    variant: Variant
    paraphrase: int = Field(ge=0)
    query_text: str
    gt_answer: Answer

    @model_validator(mode="after")
    def _answer_matches_variant(self):
        expected = self.variant.answer
        if self.gt_answer is not expected:
            raise ValueError(f"a {self.variant.value} query must have gt_answer {expected.value}")
        return self


class TrainRecord(Query):
    model_answer: Answer
    p_yes: float = Field(ge=0, le=1)
    p_no: float = Field(ge=0, le=1)
    wrong: int = Field(ge=0, le=1)  # detector target: 1 if model_answer != gt_answer

    @model_validator(mode="after")
    def _wrong_matches_answers(self):
        if self.wrong != int(self.model_answer is not self.gt_answer):
            raise ValueError("wrong must be 1 exactly when model_answer differs from gt_answer")
        return self


class ValidationAnswer(_Record):
    query_id: str
    p_yes: float = Field(ge=0, le=1)
    p_no: float = Field(ge=0, le=1)


class Metrics(_Record):
    """One row of the comparison: final answers vs the ground truth and vs the baseline answers."""

    n: int
    accuracy: float
    baseline_accuracy: float
    fixed: int  # wrong at baseline, right now
    broken: int  # right at baseline, wrong now
    net_gain: int
    changed_rate: float  # share of answers that differ from the baseline's
    mean_confidence: float  # mean of max(p_yes, p_no)
    accuracy_by_label: dict[Predicate, float]
    flagged_rate: float | None = None  # share of questions the method acted on
    flag_precision: float | None = None  # share of those that were wrong at baseline


class TuneResult(_Record):
    """The setting chosen on the validation split for one method: a threshold (feedback) or alpha (steering)."""

    threshold: float | None = None
    alpha: float | None = None
    val: Metrics
    grid: list[dict[str, float]]  # every value tried, with its validation accuracy

    @model_validator(mode="after")
    def _one_setting(self):
        if (self.threshold is None) == (self.alpha is None):
            raise ValueError("exactly one of threshold and alpha must be set")
        return self


class ValidationSettings(_Record):
    linear_feedback: TuneResult
    mlp_feedback: TuneResult
    linear_steering: TuneResult
    mlp_steering: TuneResult


class TestResults(_Record):
    settings: ValidationSettings
    results: dict[str, Metrics]  # row name -> metrics, baseline first
