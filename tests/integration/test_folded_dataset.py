"""Generating a dataset whose sheets have folded corners.

    ===== ==========================================================
    Class What it establishes
    ===== ==========================================================
    A     With folding off, nothing whatever changes.
    B     Folded sheets are produced, recorded and reproducible.
    C     The logical ground truth is untouched by folding.
    D     Attendance and reconciliation are untouched by folding.
    E     Folding works in the reference-scan mode too.
    F     Colour, grayscale and black-and-white all fold.
    G     The engine meets a folded sheet, and the dataset stays honest.
    ===== ==========================================================

The claim being tested, in one line:
    A fold changes the *paper*. It must change nothing else - not the answers,
    not the candidate, not the roster, and not a single pixel of a dataset that
    did not ask for folds.

Why the engine is run at all here:
    Because the interesting question about a folded sheet is what the engine
    does with it, and because a fold drawn in the wrong place would still look
    plausible. A sheet whose fold reaches no marker must still read perfectly;
    if it did not, the fold would be doing something other than what it claims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
from tests.conftest import build_answer_sheet_template

from omr_scanner.evaluation.fold_plans import (
    NO_FOLDS,
    FoldCoverage,
    FoldPolicy,
    classify,
    marker_outlines,
    measure_overlaps,
)
from omr_scanner.evaluation.ground_truth import (
    load_ground_truth_directory,
    load_manifest,
)
from omr_scanner.evaluation.synthetic_dataset import (
    GROUND_TRUTH_DIRNAME,
    IMAGES_DIRNAME,
    MANIFEST_FILENAME,
    ColorMode,
    DatasetProfile,
    RenderMode,
    generate_dataset,
    page_render_size,
    sheet_spec_from_template,
)
from omr_scanner.evaluation.test_cases import TestCaseTag
from omr_scanner.imaging.folds import FoldCorner, FoldSeverity
from omr_scanner.imaging.synthetic import render_sheet
from omr_scanner.services.recognition_service import RecognitionEngine
from omr_scanner.services.recognition_settings import RecognitionOptions

if TYPE_CHECKING:
    from pathlib import Path

SEED = 4242


@pytest.fixture(scope="module")
def template():
    return build_answer_sheet_template()


@pytest.fixture(scope="module")
def engine():
    return RecognitionEngine(RecognitionOptions(with_preview=False))


@pytest.fixture(scope="module")
def outlines(template):
    return marker_outlines(
        template,
        width=float(template.page.canonical_width_px),
        height=float(template.page.canonical_height_px),
    )


@pytest.fixture(scope="module")
def blank_scan(tmp_path_factory, template) -> Path:
    """A stand-in scan of the blank printed form, for the reference mode."""
    size = page_render_size(template, 300)
    sheet = render_sheet(sheet_spec_from_template(template, {}, render=size))
    path = tmp_path_factory.mktemp("reference") / "blank_form.png"
    cv2.imwrite(str(path), sheet.image)
    return path


def generate(output: Path, template, **kwargs: object):
    """A small dataset, with whatever fold policy the caller wants."""
    return generate_dataset(
        output,
        template,
        count=kwargs.pop("count", 12),
        seed=kwargs.pop("seed", SEED),
        profile=kwargs.pop("profile", DatasetProfile.BASELINE),
        write_metadata=kwargs.pop("write_metadata", False),
        **kwargs,
    )


def cohort():
    """A roster with one of every deliberate reconciliation conflict in it.

    A function rather than a fixture because two tests generate it twice - once
    per rendering they are comparing - and a population must be planned freshly
    each time to prove the plan itself does not depend on folding.
    """
    from omr_scanner.evaluation.attendance_dataset import plan_population

    return plan_population(
        count=24, set_codes=("10", "11"), seed=99, include_edge_cases=True
    )


def folded_truths(output: Path) -> dict:
    """Ground truth for the sheets that were actually folded."""
    return {
        stem: truth
        for stem, truth in load_ground_truth_directory(
            output / GROUND_TRUTH_DIRNAME
        ).items()
        if "physical_augmentation" in truth.metadata
    }


# ----------------------------------------------------------------------
# A. Off by default, and off means off
# ----------------------------------------------------------------------
class TestFoldingIsOptional:
    def test_it_is_off_unless_asked_for(self):
        assert NO_FOLDS.enabled is False
        assert FoldPolicy().enabled is False
        assert NO_FOLDS.active is False

    def test_the_default_run_folds_nothing(self, tmp_path: Path, template):
        output = tmp_path / "plain"
        generate(output, template)
        assert folded_truths(output) == {}

    def test_a_disabled_policy_is_byte_identical_to_no_policy(
        self, tmp_path: Path, template
    ):
        """The backwards-compatibility promise, at the level of the pixels.

        Not "looks the same" and not "reads the same": the same bytes. A new
        option that changed existing output even slightly would invalidate
        every baseline the benchmark has.
        """
        plain = tmp_path / "plain"
        disabled = tmp_path / "disabled"
        generate(plain, template, count=5)
        generate(
            disabled,
            template,
            count=5,
            fold_policy=FoldPolicy(enabled=False, frequency=0.9),
        )
        for index in range(1, 6):
            name = f"SYN_{index:06d}.png"
            assert (plain / IMAGES_DIRNAME / name).read_bytes() == (
                disabled / IMAGES_DIRNAME / name
            ).read_bytes(), name

    def test_a_disabled_policy_adds_no_metadata(self, tmp_path: Path, template):
        output = tmp_path / "disabled"
        generate(output, template, count=3, fold_policy=FoldPolicy(enabled=False))
        for truth in load_ground_truth_directory(
            output / GROUND_TRUTH_DIRNAME
        ).values():
            assert "physical_augmentation" not in truth.metadata

    def test_the_manifest_says_nothing_was_folded(self, tmp_path: Path, template):
        output = tmp_path / "plain"
        generate(output, template, count=2, write_metadata=True)
        assert load_manifest(output / MANIFEST_FILENAME).generator["folds"] is None


# ----------------------------------------------------------------------
# B. Folded sheets are produced and described
# ----------------------------------------------------------------------
class TestFoldedSheetsAreProduced:
    def test_some_sheets_are_folded(self, tmp_path: Path, template):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=24,
            fold_policy=FoldPolicy(enabled=True, frequency=0.3),
        )
        folded = folded_truths(output)
        assert folded
        assert len(folded) < 24, "folding everything would not be a frequency"

    def test_a_folded_sheet_differs_from_its_unfolded_self(
        self, tmp_path: Path, template
    ):
        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        generate(plain, template, count=8)
        generate(
            folded,
            template,
            count=8,
            fold_policy=FoldPolicy(enabled=True, frequency=1.0),
        )
        differing = [
            name
            for name in (f"SYN_{index:06d}.png" for index in range(1, 9))
            if (plain / IMAGES_DIRNAME / name).read_bytes()
            != (folded / IMAGES_DIRNAME / name).read_bytes()
        ]
        assert len(differing) == 8

    def test_the_record_names_the_corner_and_the_markers(
        self, tmp_path: Path, template
    ):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=20,
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        for stem, truth in folded_truths(output).items():
            record = truth.metadata["physical_augmentation"]
            assert record["corner_folds"], stem
            for entry in record["corner_folds"]:
                assert entry["corner"] in {corner.value for corner in FoldCorner}
                assert entry["severity"] in {s.value for s in FoldSeverity}
                assert entry["depth_x"] > 0.0
                assert entry["depth_y"] > 0.0
                for affected in entry["affected_markers"]:
                    assert 0.0 <= affected["overlap_fraction"] <= 1.0

    def test_the_manifest_records_the_policy(self, tmp_path: Path, template):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=6,
            write_metadata=True,
            fold_policy=FoldPolicy(
                enabled=True,
                frequency=0.5,
                corners=(FoldCorner.TOP_LEFT,),
                severity=FoldSeverity.SMALL,
            ),
        )
        policy = load_manifest(output / MANIFEST_FILENAME).generator["folds"]
        assert policy["enabled"] is True
        assert policy["corners"] == ["top_left"]
        assert policy["severity"] == "small"

    def test_generation_is_reproducible(self, tmp_path: Path, template):
        first = tmp_path / "one"
        second = tmp_path / "two"
        policy = FoldPolicy(enabled=True, frequency=0.5)
        generate(first, template, count=10, fold_policy=policy)
        generate(second, template, count=10, fold_policy=policy)
        for index in range(1, 11):
            name = f"SYN_{index:06d}.png"
            assert (first / IMAGES_DIRNAME / name).read_bytes() == (
                second / IMAGES_DIRNAME / name
            ).read_bytes(), name

    def test_the_tags_say_what_was_folded(self, tmp_path: Path, template):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=24,
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        every = {tag for truth in folded_truths(output).values() for tag in truth.tags}
        severities = {
            TestCaseTag.FOLD_MICRO.value,
            TestCaseTag.FOLD_SMALL.value,
            TestCaseTag.FOLD_MODERATE.value,
            TestCaseTag.FOLD_SEVERE.value,
        }
        assert every & severities


# ----------------------------------------------------------------------
# C. The logical truth is untouched
# ----------------------------------------------------------------------
class TestTheAnswersAreUnchanged:
    def test_folding_changes_no_answer_no_roll_no_set(self, tmp_path: Path, template):
        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        generate(plain, template, count=16)
        generate(
            folded,
            template,
            count=16,
            fold_policy=FoldPolicy(enabled=True, frequency=1.0),
        )
        left = load_ground_truth_directory(plain / GROUND_TRUTH_DIRNAME)
        right = load_ground_truth_directory(folded / GROUND_TRUTH_DIRNAME)
        assert set(left) == set(right)
        for stem in left:
            assert left[stem].roll == right[stem].roll, stem
            assert left[stem].set_code == right[stem].set_code, stem
            assert left[stem].answers == right[stem].answers, stem
            assert left[stem].ambiguous == right[stem].ambiguous, stem

    def test_the_degradation_record_is_unchanged(self, tmp_path: Path, template):
        # A fold is physical damage, not a scanner setting; it belongs in its
        # own record rather than mixed into the distortion parameters.
        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        generate(plain, template, count=8)
        generate(
            folded, template, count=8, fold_policy=FoldPolicy(enabled=True, frequency=1.0)
        )
        left = load_ground_truth_directory(plain / GROUND_TRUTH_DIRNAME)
        right = load_ground_truth_directory(folded / GROUND_TRUTH_DIRNAME)
        for stem in left:
            assert left[stem].degradation == right[stem].degradation, stem


# ----------------------------------------------------------------------
# D. Attendance survives folding
# ----------------------------------------------------------------------
class TestAttendanceSurvivesFolding:
    def test_the_roster_is_identical(self, tmp_path: Path, template):
        from omr_scanner.evaluation.attendance_dataset import (
            CANDIDATES_FILENAME,
            RECONCILIATION_FILENAME,
        )

        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        generate(plain, template, population=cohort(), write_metadata=True)
        generate(
            folded,
            template,
            population=cohort(),
            write_metadata=True,
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        for name in (CANDIDATES_FILENAME, RECONCILIATION_FILENAME):
            assert (plain / GROUND_TRUTH_DIRNAME / name).read_bytes() == (
                folded / GROUND_TRUTH_DIRNAME / name
            ).read_bytes(), name

    def test_folding_does_not_change_who_a_sheet_belongs_to(
        self, tmp_path: Path, template
    ):
        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        generate(plain, template, population=cohort())
        generate(
            folded,
            template,
            population=cohort(),
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        left = load_ground_truth_directory(plain / GROUND_TRUTH_DIRNAME)
        right = load_ground_truth_directory(folded / GROUND_TRUTH_DIRNAME)
        for stem in left:
            assert left[stem].roll == right[stem].roll, stem
            assert left[stem].duplicate_group == right[stem].duplicate_group, stem

    def test_folded_sheets_appear_in_a_roster_run(self, tmp_path: Path, template):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            population=cohort(),
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        assert folded_truths(output)


# ----------------------------------------------------------------------
# E. The reference-scan mode folds too
# ----------------------------------------------------------------------
class TestFoldingARealScan:
    def test_sheets_are_folded_after_the_marks_are_composited(
        self, tmp_path: Path, template, blank_scan: Path
    ):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=6,
            render_mode=RenderMode.REFERENCE_SCAN,
            reference_scan=blank_scan,
            fold_policy=FoldPolicy(enabled=True, frequency=1.0),
        )
        assert len(folded_truths(output)) == 6

    def test_the_folded_scan_differs_from_the_unfolded_one(
        self, tmp_path: Path, template, blank_scan: Path
    ):
        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        always = FoldPolicy(enabled=True, frequency=1.0)
        for output, policy in ((plain, NO_FOLDS), (folded, always)):
            generate(
                output,
                template,
                count=2,
                render_mode=RenderMode.REFERENCE_SCAN,
                reference_scan=blank_scan,
                fold_policy=policy,
            )
        name = "SYN_000001.png"
        first = cv2.imread(str(plain / IMAGES_DIRNAME / name), cv2.IMREAD_UNCHANGED)
        second = cv2.imread(str(folded / IMAGES_DIRNAME / name), cv2.IMREAD_UNCHANGED)
        assert first.shape == second.shape
        assert not np.array_equal(first, second)

    def test_the_fold_lands_on_the_paper_not_the_image_corner(
        self, tmp_path: Path, template, blank_scan: Path
    ):
        """The coordinate-space claim, checked where it could go wrong.

        In this mode the page sits inside a larger scan. A fold computed in
        image coordinates would appear in the margin; it has to follow the
        registration transform onto the paper's own corner.
        """
        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        policy = FoldPolicy(
            enabled=True,
            frequency=1.0,
            corners=(FoldCorner.TOP_LEFT,),
            severity=FoldSeverity.SEVERE,
        )
        for output, chosen in ((plain, NO_FOLDS), (folded, policy)):
            generate(
                output,
                template,
                count=1,
                render_mode=RenderMode.REFERENCE_SCAN,
                reference_scan=blank_scan,
                fold_policy=chosen,
            )
        name = "SYN_000001.png"
        first = cv2.imread(str(plain / IMAGES_DIRNAME / name), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(folded / IMAGES_DIRNAME / name), cv2.IMREAD_GRAYSCALE)
        changed = np.argwhere(first != second)
        assert changed.size > 0
        # In the top-left quadrant, and nowhere near the opposite corner.
        assert changed[:, 0].max() < first.shape[0] * 0.4
        assert changed[:, 1].max() < first.shape[1] * 0.4

    def test_the_rest_of_the_scan_is_untouched(
        self, tmp_path: Path, template, blank_scan: Path
    ):
        plain = tmp_path / "plain"
        folded = tmp_path / "folded"
        policy = FoldPolicy(
            enabled=True, frequency=1.0, corners=(FoldCorner.TOP_LEFT,)
        )
        for output, chosen in ((plain, NO_FOLDS), (folded, policy)):
            generate(
                output,
                template,
                count=1,
                render_mode=RenderMode.REFERENCE_SCAN,
                reference_scan=blank_scan,
                fold_policy=chosen,
            )
        name = "SYN_000001.png"
        first = cv2.imread(str(plain / IMAGES_DIRNAME / name), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(folded / IMAGES_DIRNAME / name), cv2.IMREAD_GRAYSCALE)
        height, width = first.shape
        middle = (slice(height // 2, 3 * height // 4), slice(width // 2, 3 * width // 4))
        assert np.array_equal(first[middle], second[middle])

    def test_the_ground_truth_is_the_same_in_both_modes(
        self, tmp_path: Path, template, blank_scan: Path
    ):
        drawn = tmp_path / "drawn"
        laid = tmp_path / "laid"
        policy = FoldPolicy(enabled=True, frequency=0.5)
        generate(drawn, template, count=12, fold_policy=policy)
        generate(
            laid,
            template,
            count=12,
            render_mode=RenderMode.REFERENCE_SCAN,
            reference_scan=blank_scan,
            fold_policy=policy,
        )
        left = load_ground_truth_directory(drawn / GROUND_TRUTH_DIRNAME)
        right = load_ground_truth_directory(laid / GROUND_TRUTH_DIRNAME)
        for stem in left:
            assert left[stem].answers == right[stem].answers, stem
            # The same sheets are folded, with the same folds, because folding
            # is planned from the seed before either mode draws anything.
            assert (
                "physical_augmentation" in left[stem].metadata
            ) == ("physical_augmentation" in right[stem].metadata), stem
            if "physical_augmentation" in left[stem].metadata:
                assert (
                    left[stem].metadata["physical_augmentation"]
                    == right[stem].metadata["physical_augmentation"]
                ), stem


# ----------------------------------------------------------------------
# F. Colour
# ----------------------------------------------------------------------
class TestColourWithFolds:
    @pytest.mark.parametrize(
        ("mode", "channels"),
        [
            (ColorMode.GRAYSCALE, 1),
            (ColorMode.COLOR, 3),
            (ColorMode.BLACK_AND_WHITE, 1),
        ],
    )
    def test_a_folded_sheet_comes_out_in_the_right_layout(
        self, tmp_path: Path, template, mode, channels
    ):
        output = tmp_path / f"folded_{mode.value}"
        generate(
            output,
            template,
            count=1,
            color_mode=mode,
            fold_policy=FoldPolicy(
                enabled=True, frequency=1.0, severity=FoldSeverity.SEVERE
            ),
        )
        image = cv2.imread(
            str(output / IMAGES_DIRNAME / "SYN_000001.png"), cv2.IMREAD_UNCHANGED
        )
        assert (1 if image.ndim == 2 else image.shape[2]) == channels

    def test_black_and_white_stays_two_valued_through_a_fold(
        self, tmp_path: Path, template
    ):
        output = tmp_path / "bw"
        generate(
            output,
            template,
            count=1,
            color_mode=ColorMode.BLACK_AND_WHITE,
            fold_policy=FoldPolicy(
                enabled=True, frequency=1.0, severity=FoldSeverity.SEVERE
            ),
        )
        image = cv2.imread(
            str(output / IMAGES_DIRNAME / "SYN_000001.png"), cv2.IMREAD_UNCHANGED
        )
        assert set(np.unique(image).tolist()) <= {0, 255}


# ----------------------------------------------------------------------
# G. What the engine makes of it
# ----------------------------------------------------------------------
class TestTheEngineMeetsAFoldedSheet:
    def test_a_fold_that_reaches_no_marker_still_reads_perfectly(
        self, tmp_path: Path, template, engine
    ):
        """The case that must not regress.

        Most real folds are small dog-ears nowhere near a marker. A sheet like
        that is an ordinary sheet, and if the engine started failing them the
        fold model would be doing more than folding a corner.
        """
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=24,
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )

        checked = 0
        for truth in folded_truths(output).values():
            record = truth.metadata["physical_augmentation"]
            if record["marker_interaction"] != FoldCoverage.NO_MARKER.value:
                continue
            result = engine.process(output / IMAGES_DIRNAME / truth.scan, template)
            assert result.identifier_value == truth.roll, truth.scan
            for number, expected in truth.answers.items():
                assert result.answer(number).value == expected, f"{truth.scan} q{number}"
            checked += 1
        # Guards the loop: "every marker-free fold read perfectly" is trivially
        # true of none at all.
        assert checked, "no marker-free fold was generated to check"

    def test_a_sheet_whose_marker_is_gone_is_expected_to_fail(
        self, tmp_path: Path, template, outlines
    ):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=24,
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        lost = [
            truth
            for truth in folded_truths(output).values()
            if truth.metadata["physical_augmentation"]["registration_marker_lost"]
        ]
        assert lost, "no fold covered a registration marker"
        for truth in lost:
            assert truth.expect_failure, truth.scan

    def test_a_partly_covered_marker_asserts_no_outcome(
        self, tmp_path: Path, template
    ):
        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=24,
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        partial = [
            truth
            for truth in folded_truths(output).values()
            if not truth.metadata["physical_augmentation"]["registration_marker_lost"]
        ]
        assert partial
        for truth in partial:
            assert not truth.expect_failure, truth.scan

    def test_the_recorded_interaction_matches_the_planned_folds(
        self, tmp_path: Path, template, outlines
    ):
        """The record and the geometry must not be able to disagree.

        Both come from the same measurement, and this is what proves it - a
        record written from the *request* rather than from the outcome would
        show up here as soon as the solver landed in a neighbouring band.
        """
        from omr_scanner.imaging.folds import FoldSpec

        output = tmp_path / "folded"
        generate(
            output,
            template,
            count=24,
            fold_policy=FoldPolicy(enabled=True, frequency=0.5),
        )
        width = float(template.page.canonical_width_px)
        height = float(template.page.canonical_height_px)

        for truth in folded_truths(output).values():
            record = truth.metadata["physical_augmentation"]
            specs = tuple(
                FoldSpec(
                    corner=FoldCorner(entry["corner"]),
                    severity=FoldSeverity(entry["severity"]),
                    depth_x=entry["depth_x"],
                    depth_y=entry["depth_y"],
                )
                for entry in record["corner_folds"]
            )
            recomputed = classify(
                measure_overlaps(specs, outlines, width=width, height=height)
            )
            assert recomputed.value == record["marker_interaction"], truth.scan
