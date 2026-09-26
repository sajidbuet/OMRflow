"""Which markers a fold hits, and which sheets get folded.

The question this file exists to answer:
    Not "does folding work" - ``tests/unit/test_folds.py`` measures that - but
    "does the dataset know what its folds did". A fold that covers a
    registration marker and a fold that misses it by a millimetre produce very
    different recognition outcomes, and a dataset that could not tell them
    apart would report a single folded-sheet accuracy number that means
    nothing.

Two templates, on purpose:
    * the **real** fixture template, whose top-left corner carries a
      registration marker *and* the orientation mark - the combination the
      brief calls out, and the one that decides whether two-marker coverage is
      reachable on a sheet anybody actually prints;
    * a **purpose-built** template with two registration marks deliberately
      side by side at one corner, so the two-marker machinery can be exercised
      without depending on the first template's geometry staying as it is.

    The honest result on the real template is asserted as what it is, measured
    - not assumed in either direction.

The rule this file enforces everywhere:
    A tag describes what was **measured**, never what was requested. A fold
    asked to cover two markers that in fact reached one is a one-marker fold,
    and must be labelled as one. That is the same tag-honesty rule Checkpoint
    A's ``strip_unrenderable_defects`` already applies.
"""

from __future__ import annotations

import json

import pytest
from tests.conftest import build_answer_sheet_template

from omr_scanner.domain.geometry import NormalizedPoint, NormalizedSize
from omr_scanner.domain.template import MarkerRole, OrientationMarker, RegistrationMarker
from omr_scanner.evaluation.case_plans import plan_dataset
from omr_scanner.evaluation.fold_plans import (
    COVERED_OVERLAP,
    NO_FOLDS,
    ORIENTATION_MARKER,
    PARTIAL_OVERLAP,
    REGISTRATION_MARKER,
    SUBSTANTIAL_OVERLAP,
    TOUCH_OVERLAP,
    FoldCoverage,
    FoldPolicy,
    classify,
    describe_folds,
    loses_registration_marker,
    marker_outlines,
    measure_overlaps,
    plan_folds,
    random_fold,
    solve_fold,
)
from omr_scanner.evaluation.synthetic_dataset import DatasetProfile
from omr_scanner.evaluation.test_cases import TestCaseTag
from omr_scanner.imaging.folds import FoldCorner, FoldSeverity, FoldSpec


@pytest.fixture(scope="module")
def template():
    return build_answer_sheet_template()


@pytest.fixture(scope="module")
def page(template) -> tuple[float, float]:
    return (
        float(template.page.canonical_width_px),
        float(template.page.canonical_height_px),
    )


@pytest.fixture(scope="module")
def outlines(template, page):
    width, height = page
    return marker_outlines(template, width=width, height=height)


@pytest.fixture(scope="module")
def crowded_template(template):
    """A template with two registration marks side by side at the top-left.

    Purpose-built, and deliberately not a claim about any real form: it exists
    so that the two-marker machinery can be exercised against geometry chosen
    for the test rather than inherited from a template that may change. Built
    by moving one existing marker rather than by adding a new kind of marker,
    because the fold code must work from *positions* and knows nothing about
    what a marker is called.
    """
    markers = []
    for marker in template.registration_markers:
        if marker.role is MarkerRole.TOP_RIGHT:
            # Brought round beside the top-left one. Still a perfectly ordinary
            # RegistrationMarker; only its centre has moved.
            markers.append(
                RegistrationMarker(
                    role=marker.role,
                    # Close enough to the top-left one that a single fold
                    # covers a solid share of both, rather than clipping the
                    # second by a hair - the test should be about the
                    # machinery, not about a threshold it sits next to.
                    center=NormalizedPoint(x=0.095, y=0.045),
                    size=marker.size,
                    search_radius=marker.search_radius,
                )
            )
        else:
            markers.append(marker)
    orientation = OrientationMarker(
        center=NormalizedPoint(x=0.075, y=0.075),
        size=NormalizedSize(width=0.03, height=0.012),
        expected_near=MarkerRole.TOP_LEFT,
    )
    return template.model_copy(
        update={
            "registration_markers": tuple(markers),
            "orientation_marker": orientation,
        }
    )


def fold(corner: FoldCorner, depth_x: float, depth_y: float) -> FoldSpec:
    """A fold with explicit depths, named by the deeper of the two."""
    from omr_scanner.imaging.folds import severity_for_depth

    return FoldSpec(
        corner=corner,
        severity=severity_for_depth(max(depth_x, depth_y)),
        depth_x=depth_x,
        depth_y=depth_y,
    )


# ----------------------------------------------------------------------
# Reading the template's real geometry
# ----------------------------------------------------------------------
class TestMarkerOutlines:
    def test_every_mark_is_described(self, outlines, template):
        assert len(outlines) == len(template.registration_markers) + 1
        kinds = {outline.marker_type for outline in outlines}
        assert kinds == {REGISTRATION_MARKER, ORIENTATION_MARKER}

    def test_registration_markers_keep_their_corner_names(self, outlines):
        registration = {
            outline.marker_id
            for outline in outlines
            if outline.marker_type == REGISTRATION_MARKER
        }
        assert registration == {role.value for role in MarkerRole}

    def test_the_orientation_mark_is_included(self, outlines):
        """It is as losable as any other, and often the first one a fold meets."""
        assert any(outline.marker_id == ORIENTATION_MARKER for outline in outlines)

    def test_each_outline_is_a_rectangle_at_the_declared_place(
        self, outlines, template, page
    ):
        width, height = page
        marker = next(
            item
            for item in template.registration_markers
            if item.role is MarkerRole.TOP_LEFT
        )
        outline = next(
            item for item in outlines if item.marker_id == MarkerRole.TOP_LEFT.value
        )
        xs = [point.x for point in outline.polygon]
        ys = [point.y for point in outline.polygon]
        assert len(outline.polygon) == 4
        assert (min(xs) + max(xs)) / 2 == pytest.approx(marker.center.x * width)
        assert (min(ys) + max(ys)) / 2 == pytest.approx(marker.center.y * height)
        assert max(xs) - min(xs) == pytest.approx(marker.size.width * width)

    def test_nothing_assumes_a_marker_sits_in_a_corner(self, crowded_template, page):
        # The whole design rests on this: a marker moved well along an edge is
        # still just a polygon, with no new "side marker" concept anywhere.
        width, height = page
        moved = marker_outlines(crowded_template, width=width, height=height)
        top_right = next(
            item for item in moved if item.marker_id == MarkerRole.TOP_RIGHT.value
        )
        assert max(point.x for point in top_right.polygon) < 0.2 * width


# ----------------------------------------------------------------------
# Measuring and classifying what a fold hit
# ----------------------------------------------------------------------
class TestMeasuringOverlap:
    def test_a_fold_that_reaches_nothing_reports_nothing(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.004, 0.004),)
        assert measure_overlaps(specs, outlines, width=width, height=height) == ()

    def test_a_deep_fold_covers_its_own_corner_marker(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.10, 0.08),)
        overlaps = measure_overlaps(specs, outlines, width=width, height=height)
        assert overlaps[0].marker_id == MarkerRole.TOP_LEFT.value
        assert overlaps[0].overlap_fraction >= COVERED_OVERLAP

    def test_markers_at_other_corners_are_unaffected(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.11, 0.11),)
        touched = {
            item.marker_id
            for item in measure_overlaps(specs, outlines, width=width, height=height)
        }
        assert MarkerRole.BOTTOM_RIGHT.value not in touched
        assert MarkerRole.BOTTOM_LEFT.value not in touched
        assert MarkerRole.TOP_RIGHT.value not in touched

    def test_results_are_strongest_first(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.10, 0.11),)
        overlaps = measure_overlaps(specs, outlines, width=width, height=height)
        fractions = [item.overlap_fraction for item in overlaps]
        assert fractions == sorted(fractions, reverse=True)

    def test_several_folds_take_each_marker_s_largest_overlap(self, outlines, page):
        # Never the sum: two folds cannot both cover the same marker, and
        # adding them would double-count any region they shared.
        width, height = page
        specs = (
            fold(FoldCorner.TOP_LEFT, 0.10, 0.08),
            fold(FoldCorner.BOTTOM_RIGHT, 0.10, 0.08),
        )
        overlaps = measure_overlaps(specs, outlines, width=width, height=height)
        assert all(item.overlap_fraction <= 1.0 for item in overlaps)
        touched = {item.marker_id for item in overlaps}
        assert MarkerRole.TOP_LEFT.value in touched
        assert MarkerRole.BOTTOM_RIGHT.value in touched

    def test_a_graze_below_the_floor_is_not_reported(self, outlines, page):
        """Otherwise every fold would claim to have touched something.

        Polygon intersection returns a few millionths for a fold that merely
        reaches a marker's corner, and a category populated by rounding error
        is worse than no category.
        """
        width, height = page
        for item in measure_overlaps(
            (fold(FoldCorner.TOP_LEFT, 0.06, 0.05),),
            outlines,
            width=width,
            height=height,
        ):
            assert item.overlap_fraction >= TOUCH_OVERLAP


class TestClassification:
    def test_nothing_touched(self):
        assert classify(()) is FoldCoverage.NO_MARKER

    @pytest.mark.parametrize(
        ("fraction", "expected"),
        [
            (TOUCH_OVERLAP, FoldCoverage.MARKER_TOUCH),
            (PARTIAL_OVERLAP, FoldCoverage.PARTIAL_MARKER),
            (SUBSTANTIAL_OVERLAP, FoldCoverage.SUBSTANTIAL_MARKER),
            (COVERED_OVERLAP, FoldCoverage.FULL_MARKER),
            (1.0, FoldCoverage.FULL_MARKER),
        ],
    )
    def test_one_marker_at_each_band(self, fraction, expected):
        from omr_scanner.evaluation.fold_plans import MarkerOverlap

        overlaps = (
            MarkerOverlap(
                marker_id="top_left",
                marker_type=REGISTRATION_MARKER,
                overlap_fraction=fraction,
            ),
        )
        assert classify(overlaps) is expected

    def test_two_markers_outrank_one_deep_one(self):
        from omr_scanner.evaluation.fold_plans import MarkerOverlap

        overlaps = (
            MarkerOverlap("top_left", REGISTRATION_MARKER, 1.0),
            MarkerOverlap(ORIENTATION_MARKER, ORIENTATION_MARKER, 0.2),
        )
        assert classify(overlaps) is FoldCoverage.TWO_MARKERS

    def test_a_second_marker_merely_grazed_does_not_count_as_two(self):
        from omr_scanner.evaluation.fold_plans import MarkerOverlap

        overlaps = (
            MarkerOverlap("top_left", REGISTRATION_MARKER, 1.0),
            MarkerOverlap(ORIENTATION_MARKER, ORIENTATION_MARKER, 0.01),
        )
        assert classify(overlaps) is FoldCoverage.FULL_MARKER

    def test_only_a_registration_marker_counts_as_lost(self):
        from omr_scanner.evaluation.fold_plans import MarkerOverlap

        assert loses_registration_marker(
            (MarkerOverlap("top_left", REGISTRATION_MARKER, 1.0),)
        )
        # The orientation mark going is serious, but it is not the reason a
        # page cannot be rectified.
        assert not loses_registration_marker(
            (MarkerOverlap(ORIENTATION_MARKER, ORIENTATION_MARKER, 1.0),)
        )


# ----------------------------------------------------------------------
# Solving for a wanted interaction
# ----------------------------------------------------------------------
class TestSolvingForCoverage:
    @pytest.mark.parametrize(
        "coverage",
        [
            FoldCoverage.NO_MARKER,
            FoldCoverage.MARKER_TOUCH,
            FoldCoverage.PARTIAL_MARKER,
            FoldCoverage.SUBSTANTIAL_MARKER,
            FoldCoverage.FULL_MARKER,
        ],
    )
    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_every_single_marker_case_is_reachable_at_every_corner(
        self, outlines, page, corner, coverage
    ):
        width, height = page
        solved = solve_fold(corner, coverage, outlines, width=width, height=height)
        assert solved is not None, f"{corner.value}/{coverage.value}"
        assert solved.corner is corner
        measured = measure_overlaps(
            (solved,), outlines, width=width, height=height
        )
        assert classify(measured) is coverage

    def test_a_solved_fold_lands_near_the_middle_of_its_band(self, outlines, page):
        width, height = page
        solved = solve_fold(
            FoldCorner.TOP_LEFT,
            FoldCoverage.SUBSTANTIAL_MARKER,
            outlines,
            width=width,
            height=height,
        )
        assert solved is not None
        strongest = measure_overlaps(
            (solved,), outlines, width=width, height=height
        )[0].overlap_fraction
        assert SUBSTANTIAL_OVERLAP <= strongest < COVERED_OVERLAP

    def test_restricting_the_severity_restricts_the_search(self, outlines, page):
        width, height = page
        solved = solve_fold(
            FoldCorner.TOP_LEFT,
            FoldCoverage.NO_MARKER,
            outlines,
            width=width,
            height=height,
            severity=FoldSeverity.MICRO,
        )
        assert solved is not None
        assert solved.severity is FoldSeverity.MICRO

    def test_an_impossible_request_returns_none_rather_than_an_approximation(
        self, outlines, page
    ):
        """The "not applicable" contract, which is the point of returning None.

        A micro fold is a few millimetres deep. On this template it cannot
        reach a marker at all, let alone cover one, and the honest answer is
        that the case does not exist - not a slightly larger fold pretending
        to be micro.
        """
        width, height = page
        assert (
            solve_fold(
                FoldCorner.BOTTOM_RIGHT,
                FoldCoverage.FULL_MARKER,
                outlines,
                width=width,
                height=height,
                severity=FoldSeverity.MICRO,
            )
            is None
        )

    def test_solving_is_deterministic(self, outlines, page):
        width, height = page
        first = solve_fold(
            FoldCorner.TOP_LEFT, FoldCoverage.PARTIAL_MARKER, outlines,
            width=width, height=height,
        )
        second = solve_fold(
            FoldCorner.TOP_LEFT, FoldCoverage.PARTIAL_MARKER, outlines,
            width=width, height=height,
        )
        assert first == second


class TestTwoMarkersUnderOneFold:
    def test_the_real_template_s_top_left_pair(self, outlines, page):
        """The combination the brief singles out, measured rather than assumed.

        This fixture template puts a registration marker at the page's
        top-left and the orientation mark a little along the top edge. Whether
        one fold can reach both is a fact about that geometry, so it is
        *measured* here and the measurement is the assertion - in either
        direction. What must never happen is the generator claiming the case
        without producing it.
        """
        width, height = page
        solved = solve_fold(
            FoldCorner.TOP_LEFT,
            FoldCoverage.TWO_MARKERS,
            outlines,
            width=width,
            height=height,
        )
        if solved is None:
            pytest.skip(
                "this template's top-left geometry admits no two-marker fold; "
                "the purpose-built fixture covers the machinery"
            )
        overlaps = measure_overlaps((solved,), outlines, width=width, height=height)
        affected = {item.marker_id: item for item in overlaps}
        assert MarkerRole.TOP_LEFT.value in affected
        assert ORIENTATION_MARKER in affected
        assert affected[MarkerRole.TOP_LEFT.value].marker_type == REGISTRATION_MARKER
        assert affected[ORIENTATION_MARKER].marker_type == ORIENTATION_MARKER
        # Each reported separately, with its own fraction - never one boolean.
        assert affected[MarkerRole.TOP_LEFT.value].overlap_fraction >= PARTIAL_OVERLAP
        assert affected[ORIENTATION_MARKER].overlap_fraction >= PARTIAL_OVERLAP

    def test_the_purpose_built_template(self, crowded_template, page):
        """Two registration marks side by side, so the machinery is tested.

        Independent of whatever the real template's geometry happens to be.
        """
        width, height = page
        crowded = marker_outlines(crowded_template, width=width, height=height)
        solved = solve_fold(
            FoldCorner.TOP_LEFT,
            FoldCoverage.TWO_MARKERS,
            crowded,
            width=width,
            height=height,
        )
        assert solved is not None
        overlaps = measure_overlaps((solved,), crowded, width=width, height=height)
        assert classify(overlaps) is FoldCoverage.TWO_MARKERS
        strong = [
            item for item in overlaps if item.overlap_fraction >= PARTIAL_OVERLAP
        ]
        assert len(strong) >= 2
        # Two different markers, each with its own fraction.
        assert len({item.marker_id for item in strong}) >= 2
        assert len({item.overlap_fraction for item in strong}) >= 1

    def test_a_corner_with_only_one_marker_reports_not_applicable(
        self, outlines, page
    ):
        """Three of this template's corners carry a single marker each.

        No fold at those corners can reach two, and the generator must say so
        rather than manufacture something that looks like the case.
        """
        width, height = page
        unreachable = [
            corner
            for corner in (
                FoldCorner.TOP_RIGHT,
                FoldCorner.BOTTOM_LEFT,
                FoldCorner.BOTTOM_RIGHT,
            )
            if solve_fold(
                corner, FoldCoverage.TWO_MARKERS, outlines, width=width, height=height
            )
            is None
        ]
        assert unreachable, "expected at least one corner with a single marker"


# ----------------------------------------------------------------------
# Planning a dataset
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def cases(template):
    return plan_dataset(template, count=100, seed=7, profile=DatasetProfile.BASELINE)


def folded(planned):
    return [case for case in planned if case.folds]


class TestPlanning:
    def test_a_disabled_policy_returns_the_very_same_cases(self, cases, outlines, page):
        width, height = page
        result = plan_folds(cases, NO_FOLDS, outlines, width=width, height=height, seed=7)
        assert all(before is after for before, after in zip(cases, result, strict=True))

    def test_zero_frequency_folds_nothing(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.0),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        assert folded(result) == []

    def test_the_frequency_is_roughly_honoured(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.10),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        assert 8 <= len(folded(result)) <= 14

    def test_every_eligible_corner_appears(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.15),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        corners = {spec.corner for case in folded(result) for spec in case.folds}
        assert corners == set(FoldCorner)

    def test_only_eligible_corners_appear(self, cases, outlines, page):
        width, height = page
        wanted = (FoldCorner.TOP_LEFT, FoldCorner.BOTTOM_RIGHT)
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.3, corners=wanted),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        corners = {spec.corner for case in folded(result) for spec in case.folds}
        assert corners == set(wanted)

    def test_the_interaction_classes_are_represented(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.20),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        seen = {
            classify(measure_overlaps(case.folds, outlines, width=width, height=height))
            for case in folded(result)
        }
        assert FoldCoverage.NO_MARKER in seen
        assert FoldCoverage.FULL_MARKER in seen
        assert len(seen) >= 4

    def test_coverage_can_be_switched_off(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.05, ensure_coverage=False),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        # Purely the quota now, with no deliberate cases added on top.
        assert len(folded(result)) == 5

    def test_a_fixed_severity_is_respected(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.2, severity=FoldSeverity.SMALL),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        severities = {spec.severity for case in folded(result) for spec in case.folds}
        assert severities == {FoldSeverity.SMALL}

    def test_random_severity_mixes_the_bands(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.3, severity=None),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        severities = {spec.severity for case in folded(result) for spec in case.folds}
        assert len(severities) >= 3

    def test_planning_is_deterministic(self, cases, outlines, page):
        width, height = page
        policy = FoldPolicy(enabled=True, frequency=0.2)
        first = plan_folds(cases, policy, outlines, width=width, height=height, seed=7)
        second = plan_folds(cases, policy, outlines, width=width, height=height, seed=7)
        assert [case.folds for case in first] == [case.folds for case in second]

    def test_a_different_seed_folds_different_sheets(self, cases, outlines, page):
        width, height = page
        policy = FoldPolicy(enabled=True, frequency=0.2)
        first = plan_folds(cases, policy, outlines, width=width, height=height, seed=7)
        other = plan_folds(cases, policy, outlines, width=width, height=height, seed=8)
        assert [case.folds for case in first] != [case.folds for case in other]

    def test_the_depths_are_not_all_the_same_shape(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.3),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        lopsided = [
            spec
            for case in folded(result)
            for spec in case.folds
            if abs(spec.depth_x - spec.depth_y) > 1e-6
        ]
        assert lopsided, "every fold came out a perfect triangle"


class TestSeveralCornersPerSheet:
    def test_one_per_sheet_by_default(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=0.3),
            outlines,
            width=width,
            height=height,
            seed=7,
        )
        assert max(len(case.folds) for case in folded(result)) == 1

    def test_more_can_be_asked_for(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=1.0, max_per_sheet=4),
            outlines,
            width=width,
            height=height,
            seed=21,
        )
        counts = {len(case.folds) for case in folded(result)}
        assert max(counts) > 1
        assert max(counts) <= 4

    def test_a_sheet_never_folds_one_corner_twice(self, cases, outlines, page):
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=1.0, max_per_sheet=4),
            outlines,
            width=width,
            height=height,
            seed=21,
        )
        for case in folded(result):
            corners = [spec.corner for spec in case.folds]
            assert len(corners) == len(set(corners))

    def test_three_or_more_folds_are_capped_at_moderate(self, cases, outlines, page):
        """Otherwise four severe folds take a fifth of the page between them.

        Which stops being a set of folded corners and becomes a differently
        shaped sheet.
        """
        width, height = page
        result = plan_folds(
            cases,
            FoldPolicy(
                enabled=True, frequency=1.0, max_per_sheet=4, severity=FoldSeverity.SEVERE
            ),
            outlines,
            width=width,
            height=height,
            seed=21,
        )
        busy = [case for case in folded(result) if len(case.folds) >= 3]
        assert busy
        for case in busy:
            for spec in case.folds:
                assert spec.severity is not FoldSeverity.SEVERE

    def test_a_nonsense_maximum_is_refused(self):
        with pytest.raises(ValueError, match="max_per_sheet"):
            FoldPolicy(enabled=True, max_per_sheet=9)

    def test_a_nonsense_frequency_is_refused(self):
        with pytest.raises(ValueError, match="frequency"):
            FoldPolicy(enabled=True, frequency=2.0)

    def test_enabling_with_no_corners_is_refused(self):
        with pytest.raises(ValueError, match="no corner is eligible"):
            FoldPolicy(enabled=True, corners=())


# ----------------------------------------------------------------------
# Tags and expectations
# ----------------------------------------------------------------------
class TestTagsDescribeWhatHappened:
    def plan(self, cases, outlines, page, **kwargs: object):
        width, height = page
        return plan_folds(
            cases,
            FoldPolicy(enabled=True, frequency=kwargs.pop("frequency", 0.25), **kwargs),
            outlines,
            width=width,
            height=height,
            seed=7,
        )

    def test_a_folded_sheet_says_which_severity(self, cases, outlines, page):
        for case in folded(self.plan(cases, outlines, page)):
            severities = {
                TestCaseTag[f"FOLD_{spec.severity.value.upper()}"]
                for spec in case.folds
            }
            assert severities <= set(case.tags)

    def test_the_interaction_tag_matches_the_measurement(self, cases, outlines, page):
        width, height = page
        from omr_scanner.evaluation.fold_plans import COVERAGE_TAGS

        for case in folded(self.plan(cases, outlines, page)):
            measured = classify(
                measure_overlaps(case.folds, outlines, width=width, height=height)
            )
            assert COVERAGE_TAGS[measured] in case.tags

    def test_no_sheet_claims_two_markers_without_having_two(
        self, cases, outlines, page
    ):
        width, height = page
        for case in folded(self.plan(cases, outlines, page)):
            if TestCaseTag.FOLD_TWO_MARKERS not in case.tags:
                continue
            strong = [
                item
                for item in measure_overlaps(
                    case.folds, outlines, width=width, height=height
                )
                if item.overlap_fraction >= PARTIAL_OVERLAP
            ]
            assert len(strong) >= 2

    def test_several_corners_are_tagged_as_such(self, cases, outlines, page):
        multi = self.plan(cases, outlines, page, frequency=1.0, max_per_sheet=4)
        for case in folded(multi):
            has_tag = TestCaseTag.FOLD_MULTIPLE_CORNERS in case.tags
            assert has_tag == (len(case.folds) > 1)

    def test_a_lost_registration_marker_is_an_expected_failure(
        self, cases, outlines, page
    ):
        """The only outcome a fold is allowed to assert, and it is geometry.

        At this coverage there is nothing at the corner to detect - the same
        situation the dataset already treats as an expected refusal when a
        marker was never printed.
        """
        width, height = page
        plans = self.plan(cases, outlines, page)
        lost = [
            case
            for case in folded(plans)
            if loses_registration_marker(
                measure_overlaps(case.folds, outlines, width=width, height=height)
            )
        ]
        assert lost
        for case in lost:
            assert case.expect_failure
            assert TestCaseTag.EXPECTED_FAILURE in case.tags

    def test_a_lesser_fold_asserts_no_outcome(self, cases, outlines, page):
        """The deliberately undecided band.

        A fold covering part of a marker may or may not still register, and
        which happens is a measurement of the detector rather than a property
        of the sheet. The dataset must not pre-judge it in either direction.
        """
        width, height = page
        plans = self.plan(cases, outlines, page)
        partial = [
            case
            for case in folded(plans)
            if not loses_registration_marker(
                measure_overlaps(case.folds, outlines, width=width, height=height)
            )
        ]
        assert partial
        for case in partial:
            assert not case.expect_failure

    def test_the_note_says_what_happened(self, cases, outlines, page):
        for case in folded(self.plan(cases, outlines, page)):
            assert "folded corner" in case.notes.lower()

    def test_folding_leaves_the_answers_alone(self, cases, outlines, page):
        before = {case.index: (case.roll, case.set_code, tuple(sorted(case.answers.items())))
                  for case in cases}
        for case in self.plan(cases, outlines, page):
            assert before[case.index] == (
                case.roll,
                case.set_code,
                tuple(sorted(case.answers.items())),
            )


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------
class TestMetadata:
    def test_the_shape_of_the_record(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.10, 0.08),)
        record = describe_folds(specs, outlines, width=width, height=height)

        assert set(record) == {
            "corner_folds",
            "marker_interaction",
            "registration_marker_lost",
        }
        entry = record["corner_folds"][0]
        assert entry["corner"] == "top_left"
        assert entry["severity"] in {s.value for s in FoldSeverity}
        assert entry["depth_x"] == pytest.approx(0.10, abs=1e-4)
        assert entry["depth_y"] == pytest.approx(0.08, abs=1e-4)
        assert len(entry["crease"]) == 2
        assert entry["affected_markers"]

    def test_each_affected_marker_is_named_with_its_own_fraction(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.11, 0.11),)
        entry = describe_folds(specs, outlines, width=width, height=height)[
            "corner_folds"
        ][0]
        for affected in entry["affected_markers"]:
            assert set(affected) == {"marker_id", "marker_type", "overlap_fraction"}
            assert 0.0 <= affected["overlap_fraction"] <= 1.0
            assert affected["marker_type"] in {REGISTRATION_MARKER, ORIENTATION_MARKER}

    def test_the_crease_is_recorded_normalised(self, outlines, page):
        # So the record means the same thing at any rendering resolution.
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.10, 0.08),)
        crease_points = describe_folds(specs, outlines, width=width, height=height)[
            "corner_folds"
        ][0]["crease"]
        for x, y in crease_points:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0
        assert crease_points[0][0] == pytest.approx(0.10, abs=1e-4)
        assert crease_points[1][1] == pytest.approx(0.08, abs=1e-4)

    def test_several_folds_each_get_an_entry(self, outlines, page):
        width, height = page
        specs = (
            fold(FoldCorner.TOP_LEFT, 0.06, 0.05),
            fold(FoldCorner.BOTTOM_RIGHT, 0.04, 0.03),
        )
        record = describe_folds(specs, outlines, width=width, height=height)
        assert [entry["corner"] for entry in record["corner_folds"]] == [
            "top_left",
            "bottom_right",
        ]

    def test_a_fold_that_hit_nothing_says_so(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.004, 0.004),)
        record = describe_folds(specs, outlines, width=width, height=height)
        assert record["corner_folds"][0]["affected_markers"] == []
        assert record["marker_interaction"] == FoldCoverage.NO_MARKER.value
        assert record["registration_marker_lost"] is False

    def test_it_serialises(self, outlines, page):
        width, height = page
        specs = (fold(FoldCorner.TOP_LEFT, 0.10, 0.08),)
        record = describe_folds(specs, outlines, width=width, height=height)
        assert json.loads(json.dumps(record)) == record


class TestRandomFold:
    @pytest.mark.parametrize("severity", list(FoldSeverity))
    def test_it_stays_inside_its_band(self, severity):
        import random

        from omr_scanner.imaging.folds import SEVERITY_DEPTH_RANGES

        low, high = SEVERITY_DEPTH_RANGES[severity]
        rng = random.Random(4)
        for _ in range(20):
            spec = random_fold(FoldCorner.TOP_LEFT, severity, rng)
            assert low <= spec.depth_x <= high
            assert low <= spec.depth_y <= high
            assert spec.severity is severity

    def test_the_same_generator_state_gives_the_same_fold(self):
        import random

        first = random_fold(FoldCorner.TOP_LEFT, FoldSeverity.SMALL, random.Random(5))
        second = random_fold(FoldCorner.TOP_LEFT, FoldSeverity.SMALL, random.Random(5))
        assert first == second


def test_the_policy_describes_itself_for_the_manifest():
    policy = FoldPolicy(
        enabled=True,
        corners=(FoldCorner.TOP_LEFT,),
        frequency=0.2,
        severity=FoldSeverity.MODERATE,
        max_per_sheet=2,
    )
    described = policy.describe()
    assert described["enabled"] is True
    assert described["corners"] == ["top_left"]
    assert described["severity"] == "moderate"
    assert json.loads(json.dumps(described)) == described


def test_a_random_severity_policy_says_random_rather_than_null():
    assert FoldPolicy(enabled=True).describe()["severity"] == "random"
