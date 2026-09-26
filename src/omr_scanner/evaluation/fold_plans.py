"""Deciding which sheets get folded, how hard, and what that hits.

Purpose:
    :mod:`omr_scanner.imaging.folds` knows how to fold a rectangle. It does not
    know what a registration marker is, and must not. This module is the seam:
    it reads the *template's actual geometry*, works out what a given fold
    would cover, and chooses folds that between them cover the range of cases
    worth testing.

Responsibilities:
    * :class:`FoldPolicy` - the operator's request: on or off, which corners,
      how often, how hard, how many per sheet.
    * :func:`marker_outlines` - every marker the template declares, as a
      polygon in page coordinates.
    * :func:`measure_overlaps` - what one fold actually covers, per marker.
    * :func:`solve_fold` - find a fold that produces a *wanted* interaction,
      or report honestly that this template's geometry does not allow it.
    * :func:`plan_folds` - assign folds across a planned dataset.

Why marker interaction is geometry and not a marker *type*:
    OMRFlow's template carries exactly four
    :class:`~omr_scanner.domain.template.RegistrationMarker` objects and one
    :class:`~omr_scanner.domain.template.OrientationMarker`. Nothing in it
    says "corner marker" or "side marker", and nothing should: a marker's
    relationship to a fold is decided by where it *is*, not by what it is
    called. So every marker becomes a polygon, and the question "is this
    marker affected" becomes "do these two polygons intersect, and by how
    much". A template whose registration mark sits well along an edge rather
    than in the corner is then handled with no new concept and no new code.

Why overlap is measured in page coordinates:
    A fraction of a marker's area is a property of the *paper*. Mapping the
    marker into scan pixels first and measuring there would give a different
    answer for the same physical fold depending on how the sheet happened to
    lie on the platen, because a homography does not preserve area ratios. The
    page is the frame in which the question has one answer.

Why quotas rather than dice:
    The same reasoning
    :mod:`omr_scanner.evaluation.attendance_dataset` already sets out for
    reconciliation conflicts. At a 10 per cent fold rate over 50 sheets,
    independent rolls would produce five folds *on average* and quite often
    none at one particular corner - so a dataset would silently omit cases it
    claims to cover. Filling a quota and shuffling gives the same expected
    composition with none of the variance.

What this module refuses to do:
    Fabricate an interaction the template cannot produce. If no fold at a
    given corner can reach two markers at once, :func:`solve_fold` returns
    ``None`` and the case is recorded as not applicable rather than
    approximated with something that looks similar.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from omr_scanner.evaluation.test_cases import SheetCase, TestCaseTag
from omr_scanner.imaging.folds import (
    CORNER_ORDER,
    MIN_DEPTH,
    SEVERITY_DEPTH_RANGES,
    FoldCorner,
    FoldSeverity,
    FoldSpec,
    fold_region,
    overlap_fraction,
    severity_for_depth,
)
from omr_scanner.imaging.models import Point

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from omr_scanner.domain.template import OmrTemplate

REGISTRATION_MARKER = "registration"
ORIENTATION_MARKER = "orientation"
"""The two kinds of mark a fold can take out of view. Plain strings, because
they are written into a JSON document and read back by things that are not
Python."""

TOUCH_OVERLAP = 0.005
"""Below this a marker counts as untouched.

Not zero: an anti-aliased polygon intersection returns a few millionths for a
fold that merely grazes a marker's corner, and reporting that as "touched"
would fill a dataset with a category nothing can act on."""

PARTIAL_OVERLAP = 0.05
SUBSTANTIAL_OVERLAP = 0.45
COVERED_OVERLAP = 0.90
"""Boundaries between the interaction classes, as a fraction of a marker's
area. Chosen to be far apart rather than precise - the point of a class is
that a benchmark can group by it, not that 0.44 and 0.46 are different."""

MARKER_LOST_OVERLAP = COVERED_OVERLAP
"""When a **registration** marker is considered gone rather than damaged.

The one place a fold is allowed to set ``expect_failure``, and it is a
*geometric* statement, not a calibration one: at this coverage the marker is
under the flap and there is nothing at its corner to detect, exactly as if it
had never been printed - which the dataset already treats as an expected
refusal (:attr:`~omr_scanner.evaluation.test_cases.TestCaseTag.MARKER_MISSING`).

Below it, the dataset deliberately asserts **nothing** about the outcome. A
fold covering a third of a marker may or may not still register, and which of
those happens is a measurement of the detector rather than a property of the
sheet. Those sheets carry their interaction tag and no expectation, so a
benchmark reports them as their own category instead of scoring them."""


class FoldCoverage(StrEnum):
    """A kind of fold/marker interaction a dataset should contain.

    Named by what a reader of a benchmark wants to ask: "how does the engine
    do when a fold just clips a marker?" is a different question from "when it
    covers one entirely", and a single folded-sheet accuracy number answers
    neither.
    """

    NO_MARKER = "no_marker"
    MARKER_TOUCH = "marker_touch"
    PARTIAL_MARKER = "partial_marker"
    SUBSTANTIAL_MARKER = "substantial_marker"
    FULL_MARKER = "full_marker"
    TWO_MARKERS = "two_markers"


COVERAGE_TAGS: dict[FoldCoverage, TestCaseTag] = {
    FoldCoverage.NO_MARKER: TestCaseTag.FOLD_NO_MARKER,
    FoldCoverage.MARKER_TOUCH: TestCaseTag.FOLD_MARKER_TOUCHED,
    FoldCoverage.PARTIAL_MARKER: TestCaseTag.FOLD_MARKER_PARTIAL,
    FoldCoverage.SUBSTANTIAL_MARKER: TestCaseTag.FOLD_MARKER_SUBSTANTIAL,
    FoldCoverage.FULL_MARKER: TestCaseTag.FOLD_MARKER_COVERED,
    FoldCoverage.TWO_MARKERS: TestCaseTag.FOLD_TWO_MARKERS,
}

SEVERITY_TAGS: dict[FoldSeverity, TestCaseTag] = {
    FoldSeverity.MICRO: TestCaseTag.FOLD_MICRO,
    FoldSeverity.SMALL: TestCaseTag.FOLD_SMALL,
    FoldSeverity.MODERATE: TestCaseTag.FOLD_MODERATE,
    FoldSeverity.SEVERE: TestCaseTag.FOLD_SEVERE,
}

_COVERAGE_TARGETS: dict[FoldCoverage, float] = {
    FoldCoverage.MARKER_TOUCH: 0.02,
    FoldCoverage.PARTIAL_MARKER: 0.25,
    FoldCoverage.SUBSTANTIAL_MARKER: 0.70,
    FoldCoverage.FULL_MARKER: 1.00,
}
"""Where in its band each solved fold aims, so a "partial" case is
convincingly partial rather than a hair over the boundary."""

_SOLVER_STEPS = 26
"""Depths tried per axis when solving for a coverage class.

A 26x26 grid over the permitted depth range is 676 polygon intersections,
which is microseconds, and it runs a few dozen times for a whole dataset
rather than once per sheet."""

_SOLVER_MAX_DEPTH = 0.14
"""Deepest fold the solver will consider, slightly beyond the severe band.

A little headroom, because reaching a second marker on some templates needs a
fold marginally deeper than the nominal ceiling, and refusing it would report
"not applicable" for a case that is in fact perfectly possible. Anything this
deep is still unambiguously a folded corner rather than a torn page."""

_MULTI_FOLD_SEVERITY_CAP = FoldSeverity.MODERATE
"""Hardest fold allowed once three or more corners are folded on one sheet.

Three severe folds would take a fifth of the page between them, which stops
being a set of folded corners and becomes a differently shaped sheet."""


@dataclass(frozen=True, slots=True)
class MarkerOutline:
    """One printed mark, as a polygon on the page.

    Attributes:
        marker_id: Stable identifier. The corner role for a registration
            marker (``"top_left"``), and ``"orientation"`` for the orientation
            mark - the names the template and the alignment result already use,
            rather than a numbering invented here.
        marker_type: :data:`REGISTRATION_MARKER` or :data:`ORIENTATION_MARKER`.
        polygon: Corners in page coordinates, clockwise.
    """

    marker_id: str
    marker_type: str
    polygon: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class MarkerOverlap:
    """How much of one marker a fold took out of view.

    Attributes:
        marker_id: Which marker.
        marker_type: What kind.
        overlap_fraction: Fraction of the marker's area inside the fold
            region, in ``[0, 1]``. A *fraction of the marker*, deliberately -
            see :func:`~omr_scanner.imaging.folds.overlap_fraction`.
    """

    marker_id: str
    marker_type: str
    overlap_fraction: float

    def describe(self) -> dict[str, Any]:
        """Return this overlap as JSON-safe plain data."""
        return {
            "marker_id": self.marker_id,
            "marker_type": self.marker_type,
            "overlap_fraction": round(self.overlap_fraction, 4),
        }


@dataclass(frozen=True, slots=True)
class FoldPolicy:
    """What the operator asked for, as plain data.

    Lives in the generation request rather than in the dialog, so the command
    line and the GUI ask for the same thing in the same words.

    Attributes:
        enabled: Off by default. Every other field is inert until this is on,
            and with it off a dataset is byte-identical to one generated before
            folds existed.
        corners: Which page corners may be folded.
        frequency: Fraction of sheets to affect, in ``[0, 1]``.
        severity: The band to draw depths from, or ``None`` for a mixture -
            which is what the dialog calls "Random".
        max_per_sheet: Upper bound on folded corners per affected sheet, 1 to
            4. More than one is uncommon in reality and the default is one.
        ensure_coverage: Spend some of the quota on deliberately chosen
            interactions - a fold that reaches no marker, one that just clips
            a marker, one that covers it - rather than leaving the spread to
            chance. See the module docstring on quotas.
    """

    enabled: bool = False
    corners: tuple[FoldCorner, ...] = CORNER_ORDER
    frequency: float = 0.10
    severity: FoldSeverity | None = None
    max_per_sheet: int = 1
    ensure_coverage: bool = True

    def __post_init__(self) -> None:
        """Reject a policy that could not be carried out."""
        if not 0.0 <= self.frequency <= 1.0:
            raise ValueError(f"frequency must lie in [0, 1]; got {self.frequency}")
        if not 1 <= self.max_per_sheet <= len(CORNER_ORDER):
            raise ValueError(
                f"max_per_sheet must lie in 1..{len(CORNER_ORDER)}; "
                f"got {self.max_per_sheet}"
            )
        if self.enabled and not self.corners:
            raise ValueError("Folding is enabled but no corner is eligible")
        unknown = set(self.corners) - set(CORNER_ORDER)
        if unknown:
            raise ValueError(f"Unknown fold corner(s): {sorted(unknown)}")

    @property
    def active(self) -> bool:
        """Whether this policy will actually fold anything."""
        return self.enabled and self.frequency > 0.0 and bool(self.corners)

    def describe(self) -> dict[str, Any]:
        """Return this policy as JSON-safe plain data, for the manifest."""
        return {
            "enabled": self.enabled,
            "corners": [corner.value for corner in self.corners],
            "frequency": self.frequency,
            "severity": self.severity.value if self.severity is not None else "random",
            "max_per_sheet": self.max_per_sheet,
            "ensure_coverage": self.ensure_coverage,
        }


NO_FOLDS = FoldPolicy()
"""A policy that folds nothing: the default everywhere.

A named constant rather than a call at each default, because a frozen policy
can safely be shared and because a dataclass field whose default is a function
call is exactly the pattern that makes a mutable default a bug elsewhere."""


# ----------------------------------------------------------------------
# Template geometry
# ----------------------------------------------------------------------
def marker_outlines(
    template: OmrTemplate, *, width: float, height: float
) -> tuple[MarkerOutline, ...]:
    """Return every mark the template declares, as a polygon on the page.

    Both kinds, because a fold does not distinguish them: the orientation mark
    is as losable as a registration marker, and on a sheet whose orientation
    mark sits beside the top-left corner it is frequently the *first* thing a
    fold reaches.

    Args:
        template: The template to read. Nothing is assumed about where its
            markers are - a template whose registration mark sits a third of
            the way along an edge is handled by the same arithmetic.
        width: Page width in the coordinate space overlap will be measured in.
        height: Page height in the same space.

    Returns:
        Registration markers in the template's own order, then the orientation
        mark.
    """
    outlines = [
        MarkerOutline(
            marker_id=marker.role.value,
            marker_type=REGISTRATION_MARKER,
            polygon=_rectangle(
                marker.center.x * width,
                marker.center.y * height,
                marker.size.width * width,
                marker.size.height * height,
            ),
        )
        for marker in template.registration_markers
    ]
    mark = template.orientation_marker
    outlines.append(
        MarkerOutline(
            marker_id=ORIENTATION_MARKER,
            marker_type=ORIENTATION_MARKER,
            polygon=_rectangle(
                mark.center.x * width,
                mark.center.y * height,
                mark.size.width * width,
                mark.size.height * height,
            ),
        )
    )
    return tuple(outlines)


def _rectangle(
    center_x: float, center_y: float, width: float, height: float
) -> tuple[Point, ...]:
    """A centred axis-aligned rectangle, clockwise from the top-left."""
    half_x, half_y = width / 2.0, height / 2.0
    return (
        Point(x=center_x - half_x, y=center_y - half_y),
        Point(x=center_x + half_x, y=center_y - half_y),
        Point(x=center_x + half_x, y=center_y + half_y),
        Point(x=center_x - half_x, y=center_y + half_y),
    )


# ----------------------------------------------------------------------
# Measuring and classifying
# ----------------------------------------------------------------------
def measure_overlaps(
    specs: Sequence[FoldSpec],
    outlines: Sequence[MarkerOutline],
    *,
    width: float,
    height: float,
) -> tuple[MarkerOverlap, ...]:
    """Return what a sheet's folds cover, per marker, strongest first.

    Several folds are combined by taking each marker's **largest** overlap
    rather than summing: two folds at opposite corners cannot both cover the
    same marker, and on the rare template where their regions did meet, adding
    the two would double-count the shared part.
    """
    strongest: dict[str, MarkerOverlap] = {}
    for spec in specs:
        region = fold_region(spec, width, height)
        for outline in outlines:
            fraction = overlap_fraction(region, outline.polygon)
            if fraction < TOUCH_OVERLAP:
                continue
            existing = strongest.get(outline.marker_id)
            if existing is None or fraction > existing.overlap_fraction:
                strongest[outline.marker_id] = MarkerOverlap(
                    marker_id=outline.marker_id,
                    marker_type=outline.marker_type,
                    overlap_fraction=fraction,
                )
    return tuple(
        sorted(
            strongest.values(),
            key=lambda item: (-item.overlap_fraction, item.marker_id),
        )
    )


def classify(overlaps: Sequence[MarkerOverlap]) -> FoldCoverage:
    """Name the interaction a set of measured overlaps represents.

    Measured, never requested. A fold asked to cover two markers that in fact
    reached one is classified as reaching one, which is the whole of the
    tag-honesty rule this project already applies to unrenderable defects.
    """
    if not overlaps:
        return FoldCoverage.NO_MARKER
    if sum(1 for item in overlaps if item.overlap_fraction >= PARTIAL_OVERLAP) >= 2:
        return FoldCoverage.TWO_MARKERS
    strongest = overlaps[0].overlap_fraction
    if strongest >= COVERED_OVERLAP:
        return FoldCoverage.FULL_MARKER
    if strongest >= SUBSTANTIAL_OVERLAP:
        return FoldCoverage.SUBSTANTIAL_MARKER
    if strongest >= PARTIAL_OVERLAP:
        return FoldCoverage.PARTIAL_MARKER
    return FoldCoverage.MARKER_TOUCH


def loses_registration_marker(overlaps: Sequence[MarkerOverlap]) -> bool:
    """Whether a registration marker is gone rather than merely clipped.

    The single question that decides ``expect_failure`` for a folded sheet -
    see :data:`MARKER_LOST_OVERLAP` for why that is the only place folding is
    allowed to assert an outcome.
    """
    return any(
        item.marker_type == REGISTRATION_MARKER
        and item.overlap_fraction >= MARKER_LOST_OVERLAP
        for item in overlaps
    )


# ----------------------------------------------------------------------
# Solving for a wanted interaction
# ----------------------------------------------------------------------
def solve_fold(
    corner: FoldCorner,
    coverage: FoldCoverage,
    outlines: Sequence[MarkerOutline],
    *,
    width: float,
    height: float,
    severity: FoldSeverity | None = None,
) -> FoldSpec | None:
    """Find a fold at ``corner`` that produces ``coverage``, if one exists.

    A grid search over the two depths independently - which is what lets it
    find the asymmetric folds that reach a marker sitting along one edge, and
    is why it is a grid rather than a single depth swept twice.

    Args:
        corner: Which corner to fold.
        coverage: The interaction wanted.
        outlines: The template's markers, in page coordinates.
        width: Page width.
        height: Page height.
        severity: Restrict the search to one severity band. ``None`` searches
            the whole permitted range.

    Returns:
        The shallowest fold whose measured coverage matches, preferring one
        whose strongest overlap sits near the middle of the wanted band, or
        ``None`` when this template's geometry admits no such fold at this
        corner. ``None`` is a real answer and must be recorded as "not
        applicable" rather than worked around.
    """
    low, high = (
        SEVERITY_DEPTH_RANGES[severity]
        if severity is not None
        else (MIN_DEPTH, _SOLVER_MAX_DEPTH)
    )
    target = _COVERAGE_TARGETS.get(coverage)
    best: tuple[float, FoldSpec] | None = None

    for step_x in range(_SOLVER_STEPS):
        depth_x = low + (high - low) * step_x / (_SOLVER_STEPS - 1)
        for step_y in range(_SOLVER_STEPS):
            depth_y = low + (high - low) * step_y / (_SOLVER_STEPS - 1)
            candidate = FoldSpec(
                corner=corner,
                # When a band was asked for, the answer is that band. Deriving
                # it from the depth instead would mislabel a fold sitting
                # exactly on a boundary - the shallowest fold in the "small"
                # band has depth 0.015, which is also the top of "micro", and a
                # dataset asked for small folds would quietly contain a micro
                # one.
                severity=(
                    severity
                    if severity is not None
                    else severity_for_depth(max(depth_x, depth_y))
                ),
                depth_x=depth_x,
                depth_y=depth_y,
            )
            overlaps = measure_overlaps(
                (candidate,), outlines, width=width, height=height
            )
            if classify(overlaps) is not coverage:
                continue
            strongest = overlaps[0].overlap_fraction if overlaps else 0.0
            # Prefer the wanted depth of coverage; break ties towards the
            # shallower fold, so a "no marker" case is a small fold rather
            # than a large one that happens to miss everything.
            score = (
                abs(strongest - target) if target is not None else 0.0
            ) + 0.01 * (depth_x + depth_y)
            if best is None or score < best[0]:
                best = (score, candidate)

    return best[1] if best is not None else None


# ----------------------------------------------------------------------
# Drawing a fold at random
# ----------------------------------------------------------------------
def random_fold(
    corner: FoldCorner,
    severity: FoldSeverity,
    rng: random.Random,
) -> FoldSpec:
    """Return one fold of the given class, with independent depths.

    The two depths are drawn separately, so a fold is almost never a 45 degree
    triangle - which matters because a dataset of identical isoceles corners
    would test one shape very thoroughly and every other shape not at all.
    """
    low, high = SEVERITY_DEPTH_RANGES[severity]
    return FoldSpec(
        corner=corner,
        severity=severity,
        depth_x=rng.uniform(low, high),
        depth_y=rng.uniform(low, high),
    )


def _severity_for(policy: FoldPolicy, rng: random.Random, count: int) -> FoldSeverity:
    """Pick a severity, honouring the policy and the multi-fold cap."""
    chosen = (
        policy.severity
        if policy.severity is not None
        else rng.choice(list(FoldSeverity))
    )
    if count >= 3 and _order(chosen) > _order(_MULTI_FOLD_SEVERITY_CAP):
        return _MULTI_FOLD_SEVERITY_CAP
    return chosen


def _order(severity: FoldSeverity) -> int:
    """Rank a severity, so two can be compared."""
    return list(FoldSeverity).index(severity)


# ----------------------------------------------------------------------
# Planning a whole dataset
# ----------------------------------------------------------------------
def plan_folds(
    cases: Sequence[SheetCase],
    policy: FoldPolicy,
    outlines: Sequence[MarkerOutline],
    *,
    width: float,
    height: float,
    seed: int,
) -> list[SheetCase]:
    """Return ``cases`` with folds assigned according to ``policy``.

    Args:
        cases: The planned dataset, before folding.
        policy: What was asked for.
        outlines: The template's markers, in the same page coordinates the
            folds will be measured in.
        width: Page width.
        height: Page height.
        seed: The dataset's master seed. Folding is decided from this alone,
            so the same request produces the same folded sheets in any process.

    Returns:
        A new list. Cases that were not selected are returned **unchanged and
        identical**, so a dataset with folding switched off - or one whose
        sheets were simply not chosen - renders exactly as it did before.

    Each selected case gains its folds, the tags describing what they measured
    (never what they were asked for), and a note. ``expect_failure`` is set
    only when a registration marker is genuinely gone; see
    :data:`MARKER_LOST_OVERLAP`.
    """
    if not policy.active or not cases:
        return list(cases)

    rng = random.Random(f"{seed}:folds")
    plans = _fold_plans(policy, outlines, width=width, height=height, rng=rng, total=len(cases))
    if not plans:
        return list(cases)

    slots: list[tuple[FoldSpec, ...] | None] = [*plans, *[None] * (len(cases) - len(plans))]
    rng.shuffle(slots)

    return [
        case if folds is None else _fold_case(case, folds, outlines, width, height)
        for case, folds in zip(cases, slots, strict=True)
    ]


def _fold_plans(
    policy: FoldPolicy,
    outlines: Sequence[MarkerOutline],
    *,
    width: float,
    height: float,
    rng: random.Random,
    total: int,
) -> list[tuple[FoldSpec, ...]]:
    """Build the list of folded sheets: coverage cases first, then the quota.

    The same shape as
    :mod:`omr_scanner.evaluation.attendance_dataset`'s conflict assignment -
    deliberate cases are emitted unconditionally, the rest of the quota is
    filled at random, and a quota too small to hold the deliberate cases keeps
    them and loses the random ones. A dataset that omits a case cannot be used
    to prove that case is handled.
    """
    quota = round(total * policy.frequency)
    plans: list[tuple[FoldSpec, ...]] = []

    if policy.ensure_coverage:
        for index, coverage in enumerate(FoldCoverage):
            corner = policy.corners[index % len(policy.corners)]
            solved = solve_fold(
                corner, coverage, outlines, width=width, height=height,
                severity=policy.severity,
            )
            if solved is not None:
                plans.append((solved,))
        # One fold at each eligible corner, whatever the coverage cases used,
        # so "all four corners" is a property of the dataset and not a hope.
        for corner in policy.corners:
            if not any(fold[0].corner is corner for fold in plans):
                plans.append((random_fold(corner, _severity_for(policy, rng, 1), rng),))

    while len(plans) < quota:
        plans.append(_random_sheet_folds(policy, rng))

    if len(plans) > total:
        plans = plans[:total]
    return plans


def _random_sheet_folds(
    policy: FoldPolicy, rng: random.Random
) -> tuple[FoldSpec, ...]:
    """One affected sheet's worth of folds, chosen at random."""
    eligible = list(policy.corners)
    count = min(rng.randint(1, policy.max_per_sheet), len(eligible))
    corners = rng.sample(eligible, count)
    return tuple(
        random_fold(corner, _severity_for(policy, rng, count), rng)
        for corner in corners
    )


def _fold_case(
    case: SheetCase,
    folds: tuple[FoldSpec, ...],
    outlines: Sequence[MarkerOutline],
    width: float,
    height: float,
) -> SheetCase:
    """Attach folds to one case, with honest tags and a note."""
    overlaps = measure_overlaps(folds, outlines, width=width, height=height)
    coverage = classify(overlaps)
    lost = loses_registration_marker(overlaps)

    tags = list(case.tags)
    for spec in folds:
        _append(tags, SEVERITY_TAGS[spec.severity])
    if len(folds) > 1:
        _append(tags, TestCaseTag.FOLD_MULTIPLE_CORNERS)
    _append(tags, COVERAGE_TAGS[coverage])
    if lost:
        _append(tags, TestCaseTag.EXPECTED_FAILURE)

    corners = ", ".join(spec.corner.value for spec in folds)
    note = (
        f"Physically folded corner(s): {corners}. "
        + (
            "A registration marker is under the fold, so this sheet is not "
            "expected to register."
            if lost
            else f"Marker interaction: {coverage.value}."
        )
    )
    return replace(
        case,
        folds=folds,
        tags=tuple(tags),
        expect_failure=case.expect_failure or lost,
        notes=f"{case.notes} {note}".strip(),
    )


def _append(tags: list[TestCaseTag], tag: TestCaseTag) -> None:
    """Add a tag once, preserving the order they were established in."""
    if tag not in tags:
        tags.append(tag)


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------
def describe_folds(
    folds: Sequence[FoldSpec],
    outlines: Sequence[MarkerOutline],
    *,
    width: float,
    height: float,
) -> dict[str, Any]:
    """Return a sheet's folds as JSON-safe plain data.

    Per fold: which corner, which severity band, both depths, the crease in
    normalised page coordinates, and **every marker it touched with that
    marker's own overlap fraction** - never one boolean for "a marker was
    affected", because which marker and how much is the whole of what a
    failure investigation needs.

    The crease is recorded normalised rather than in pixels so that the record
    means the same thing whatever resolution the sheet was rendered at.
    """
    entries: list[dict[str, Any]] = []
    for spec in folds:
        region = fold_region(spec, width, height)
        affected = [
            MarkerOverlap(
                marker_id=outline.marker_id,
                marker_type=outline.marker_type,
                overlap_fraction=fraction,
            ).describe()
            for outline in outlines
            if (fraction := overlap_fraction(region, outline.polygon)) >= TOUCH_OVERLAP
        ]
        first, second = region[1], region[3]
        entries.append(
            {
                "corner": spec.corner.value,
                "severity": spec.severity.value,
                "depth_x": round(spec.depth_x, 5),
                "depth_y": round(spec.depth_y, 5),
                "crease": [
                    [round(first.x / width, 5), round(first.y / height, 5)],
                    [round(second.x / width, 5), round(second.y / height, 5)],
                ],
                "affected_markers": sorted(
                    affected, key=lambda item: -float(item["overlap_fraction"])
                ),
            }
        )
    overlaps = measure_overlaps(folds, outlines, width=width, height=height)
    return {
        "corner_folds": entries,
        "marker_interaction": classify(overlaps).value,
        "registration_marker_lost": loses_registration_marker(overlaps),
    }


# `FoldCorner` and `FoldSeverity` are re-exported rather than merely imported:
# they are part of what an operator chooses, so the dialog and the command line
# both need them - and `docs/ARCHITECTURE.md` forbids the GUI from importing
# `omr_scanner.imaging` at all. Naming them here is what lets the choice cross
# that boundary without the algorithms following it.
__all__ = [
    "COVERAGE_TAGS",
    "COVERED_OVERLAP",
    "MARKER_LOST_OVERLAP",
    "NO_FOLDS",
    "ORIENTATION_MARKER",
    "PARTIAL_OVERLAP",
    "REGISTRATION_MARKER",
    "SEVERITY_TAGS",
    "SUBSTANTIAL_OVERLAP",
    "TOUCH_OVERLAP",
    "FoldCorner",
    "FoldCoverage",
    "FoldPolicy",
    "FoldSeverity",
    "MarkerOutline",
    "MarkerOverlap",
    "classify",
    "describe_folds",
    "loses_registration_marker",
    "marker_outlines",
    "measure_overlaps",
    "plan_folds",
    "random_fold",
    "solve_fold",
]
