"""Registering a real blank form, and putting marks where its bubbles are.

What is actually being measured:
    Every reference scan in this module is produced by distorting a known
    canonical page through a known homography, so the *right answer is known
    exactly* before registration runs. The tests then ask
    :func:`~omr_scanner.evaluation.reference_scan.load_reference_scan` to
    recover that homography and measure how far off it is, at the template's
    own bubble centres.

Why at the bubble centres and not at the markers:
    The four marker centres are the correspondences the homography is fitted
    through, so an error there is detection error alone. The bubbles are where
    a mark has to land to be read, they take no part in the fit, and the
    distance between where one ends up and where it should be is therefore the
    honest measure - the same reasoning the synthetic sheet's interior control
    points already exist for.

Why a failure here would be invisible downstream:
    A reference registered a few pixels out produces a dataset that looks
    perfect and reads slightly wrong, on every sheet, in a way no later stage
    can detect: the ground truth says B, the ink is between B and C, and the
    engine gets the blame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.conftest import build_answer_sheet_template

from omr_scanner.evaluation.reference_scan import (
    MAX_MARK_LAYER_SCALE,
    ReferenceScanError,
    load_reference_scan,
    render_onto_reference,
)
from omr_scanner.evaluation.synthetic_dataset import PageRender, sheet_spec_from_template
from omr_scanner.evaluation.test_cases import MarkPlan
from omr_scanner.imaging.models import Point
from omr_scanner.imaging.synthetic import (
    ColorMode,
    DistortionSpec,
    apply_distortion,
    project,
    render_mark_layer,
    render_sheet,
)

BUBBLE_TOLERANCE_PX = 2.0
"""How far a template bubble centre may land from where the true homography
puts it, in scan pixels.

Two pixels on a 1240x1754 page is well under a tenth of a bubble, so a mark
placed this accurately is unambiguously inside the right ring. The floor is set
by marker-centroid precision, not by arithmetic: the homography itself is
exact, and what this tolerates is how precisely four printed squares can be
located in a degraded image."""

INK_LEVEL = 60


@pytest.fixture
def template():
    return build_answer_sheet_template()


def canonical_page_render(template) -> PageRender:
    """The template's own canonical page size."""
    return PageRender(
        width=template.page.canonical_width_px,
        height=template.page.canonical_height_px,
        dpi=150,
        derived_from="canonical",
    )


def blank_sheet(template, render: PageRender | None = None):
    """The printed form with nothing marked on it."""
    size = render if render is not None else canonical_page_render(template)
    return render_sheet(sheet_spec_from_template(template, {}, render=size))


def write_reference(
    template,
    directory: Path,
    distortion: DistortionSpec,
    *,
    name: str = "blank.png",
    render: PageRender | None = None,
) -> tuple[Path, np.ndarray]:
    """Write a stand-in "real" blank scan, and return its true homography.

    The scan is a canonical page put through a known transform, which is what
    makes the right answer knowable. A genuine photograph would be more
    realistic and would tell us nothing, because nobody knows its homography.
    """
    distorted = apply_distortion(blank_sheet(template, render), distortion)
    path = directory / name
    cv2.imwrite(str(path), distorted.image)
    return path, distorted.homography


def bubble_probes(template, spec) -> tuple[Point, ...]:
    """Canonical positions of a spread of the template's bubbles."""
    return tuple(
        Point(x=spec.to_pixels(bubble.center).x, y=spec.to_pixels(bubble.center).y)
        for bubble in spec.answer_bubbles[::37]
    )


def worst_bubble_error(template, reference, true_homography) -> float:
    """Largest distance, in scan pixels, between recovered and true positions."""
    spec = sheet_spec_from_template(template, {}, render=canonical_page_render(template))
    probes = bubble_probes(template, spec)
    assert probes, "the fixture template has no bubbles to probe"
    expected = project(true_homography, probes)
    recovered = project(reference.canonical_to_scan, probes)
    return max(
        a.distance_to(b) for a, b in zip(expected, recovered, strict=True)
    )


# ----------------------------------------------------------------------
# Registration, under each thing a scanner really does to a page
# ----------------------------------------------------------------------
class TestRegistration:
    def test_identity(self, template, tmp_path):
        path, truth = write_reference(template, tmp_path, DistortionSpec(margin_px=0))
        reference = load_reference_scan(path, template)
        assert worst_bubble_error(template, reference, truth) < BUBBLE_TOLERANCE_PX

    def test_translation(self, template, tmp_path):
        path, truth = write_reference(
            template,
            tmp_path,
            DistortionSpec(margin_px=60, translate_x_px=35.0, translate_y_px=-20.0),
        )
        reference = load_reference_scan(path, template)
        assert worst_bubble_error(template, reference, truth) < BUBBLE_TOLERANCE_PX

    def test_uniform_scale(self, template, tmp_path):
        path, truth = write_reference(
            template, tmp_path, DistortionSpec(scale_x=1.35, scale_y=1.35)
        )
        reference = load_reference_scan(path, template)
        assert worst_bubble_error(template, reference, truth) < BUBBLE_TOLERANCE_PX

    def test_anisotropic_scale(self, template, tmp_path):
        # A sheet stretched along one axis - a roller feeding unevenly, or a
        # photocopier's optics. The two axes must be recovered independently;
        # a single scale factor would put every bubble in a column wrong.
        path, truth = write_reference(
            template, tmp_path, DistortionSpec(scale_x=1.15, scale_y=0.95)
        )
        reference = load_reference_scan(path, template)
        assert worst_bubble_error(template, reference, truth) < BUBBLE_TOLERANCE_PX

    def test_rotation(self, template, tmp_path):
        path, truth = write_reference(
            template, tmp_path, DistortionSpec(rotation_degrees=6.0)
        )
        reference = load_reference_scan(path, template)
        assert worst_bubble_error(template, reference, truth) < BUBBLE_TOLERANCE_PX

    def test_perspective(self, template, tmp_path):
        # The case a scale-and-offset registration cannot express at all: a
        # page photographed at an angle, where the correction is different in
        # every part of the sheet.
        path, truth = write_reference(
            template, tmp_path, DistortionSpec(perspective_strength=0.02, seed=7)
        )
        reference = load_reference_scan(path, template)
        assert worst_bubble_error(template, reference, truth) < BUBBLE_TOLERANCE_PX

    def test_a_sheet_fed_upside_down_still_registers(self, template, tmp_path):
        # Worth having: the orientation mark resolves which way up the page
        # went through the scanner, so a reference does not have to be fed the
        # right way round to be usable.
        path, truth = write_reference(
            template, tmp_path, DistortionSpec(rotation_degrees=180.0)
        )
        reference = load_reference_scan(path, template)
        assert reference.registration.quarter_turns == 2
        assert worst_bubble_error(template, reference, truth) < BUBBLE_TOLERANCE_PX

    def test_a_wrong_transform_would_be_caught(self, template, tmp_path):
        """Guards the measurement itself, not the code.

        Every test above passes when the error is small. This one proves a
        small error means something: an obviously wrong transform - the
        identity, against a page that was rotated and scaled - must be reported
        as hundreds of pixels out, not as a near miss. Without it, a
        ``worst_bubble_error`` that always returned zero would make the whole
        class pass while measuring nothing.
        """
        path, _truth = write_reference(
            template, tmp_path, DistortionSpec(rotation_degrees=6.0)
        )
        reference = load_reference_scan(path, template)
        wrong = worst_bubble_error(template, reference, np.eye(3, dtype=np.float64))
        assert wrong > 50.0 * BUBBLE_TOLERANCE_PX

    def test_the_registration_quality_is_recorded(self, template, tmp_path):
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        registration = load_reference_scan(path, template).registration
        assert registration.min_marker_score > 0.5
        assert registration.orientation_confidence > 0.5
        assert registration.max_reprojection_error_px < 1.0
        assert isinstance(registration.describe()["warnings"], list)


# ----------------------------------------------------------------------
# Refusing a file that cannot be used
# ----------------------------------------------------------------------
class TestRejection:
    def test_a_file_that_is_not_an_image(self, template, tmp_path):
        path = tmp_path / "notes.png"
        path.write_bytes(b"this is not an image")
        with pytest.raises(ReferenceScanError, match="could not be decoded"):
            load_reference_scan(path, template)

    def test_a_missing_file(self, template, tmp_path):
        with pytest.raises(ReferenceScanError, match="could not be decoded"):
            load_reference_scan(tmp_path / "absent.png", template)

    def test_a_page_with_no_registration_markers(self, template, tmp_path):
        path = tmp_path / "empty.png"
        cv2.imwrite(str(path), np.full((1754, 1240), 255, dtype=np.uint8))
        with pytest.raises(ReferenceScanError, match="does not register"):
            load_reference_scan(path, template)

    def test_the_refusal_names_the_file_and_the_template(self, template, tmp_path):
        path = tmp_path / "empty.png"
        cv2.imwrite(str(path), np.full((1754, 1240), 255, dtype=np.uint8))
        with pytest.raises(ReferenceScanError) as raised:
            load_reference_scan(path, template)
        assert "empty.png" in raised.value.user_message
        assert template.name in raised.value.user_message

    def test_a_scan_cropped_past_its_markers(self, template, tmp_path):
        path, _truth = write_reference(
            template, tmp_path, DistortionSpec(margin_px=-140)
        )
        with pytest.raises(ReferenceScanError, match="does not register"):
            load_reference_scan(path, template)


# ----------------------------------------------------------------------
# Resolution policy
# ----------------------------------------------------------------------
class TestTheMarkLayerSize:
    def test_a_canonical_sized_scan_needs_no_supersampling(self, template, tmp_path):
        path, _truth = write_reference(template, tmp_path, DistortionSpec(margin_px=0))
        reference = load_reference_scan(path, template)
        assert reference.mark_layer_scale == 1
        assert reference.mark_layer_size == (
            template.page.canonical_width_px,
            template.page.canonical_height_px,
        )

    def test_a_higher_resolution_scan_supersamples_the_layer(self, template, tmp_path):
        path, _truth = write_reference(
            template, tmp_path, DistortionSpec(margin_px=0, scale_x=2.0, scale_y=2.0)
        )
        reference = load_reference_scan(path, template)
        assert reference.mark_layer_scale == 2

    def test_supersampling_is_capped(self, template, tmp_path):
        path, _truth = write_reference(
            template, tmp_path, DistortionSpec(margin_px=0, scale_x=2.4, scale_y=2.4)
        )
        reference = load_reference_scan(
            path, template, max_mark_layer_scale=2
        )
        assert reference.mark_layer_scale == 2
        assert MAX_MARK_LAYER_SCALE >= 2

    def test_a_nonsensical_cap_is_rejected(self, template, tmp_path):
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        with pytest.raises(ValueError, match="at least 1"):
            load_reference_scan(path, template, max_mark_layer_scale=0)

    def test_the_supersampled_transform_agrees_with_the_canonical_one(
        self, template, tmp_path
    ):
        # A point at k times canonical coordinates must land exactly where the
        # canonical point does, or every mark is displaced by the scale factor.
        path, _truth = write_reference(
            template, tmp_path, DistortionSpec(margin_px=0, scale_x=2.0, scale_y=2.0)
        )
        reference = load_reference_scan(path, template)
        scale = reference.mark_layer_scale
        canonical = (Point(x=300.0, y=700.0), Point(x=900.0, y=1500.0))
        scaled = tuple(Point(x=p.x * scale, y=p.y * scale) for p in canonical)
        for expected, actual in zip(
            project(reference.canonical_to_scan, canonical),
            project(reference.mark_layer_to_scan, scaled),
            strict=True,
        ):
            assert actual.x == pytest.approx(expected.x, abs=1e-6)
            assert actual.y == pytest.approx(expected.y, abs=1e-6)


# ----------------------------------------------------------------------
# Compositing onto the registered scan
# ----------------------------------------------------------------------
class TestRenderingOntoTheReference:
    def test_a_mark_lands_in_its_bubble(self, template, tmp_path):
        path, truth = write_reference(
            template, tmp_path, DistortionSpec(rotation_degrees=4.0, scale_x=1.2, scale_y=1.2)
        )
        reference = load_reference_scan(path, template)

        layer_spec = sheet_spec_from_template(
            template,
            {"questions_0": {0: MarkPlan(labels=("B",))}},
            render=PageRender(
                width=reference.mark_layer_size[0],
                height=reference.mark_layer_size[1],
                dpi=0,
                derived_from="reference_scan",
            ),
        )
        marked = next(
            bubble for bubble in layer_spec.answer_bubbles if bubble.fill > 0.0
        )
        composited = render_onto_reference(reference, render_mark_layer(layer_spec))

        # Where the bubble *truly* is in the scan, from the homography that
        # made the scan - never from the one under test.
        canonical = sheet_spec_from_template(
            template, {}, render=canonical_page_render(template)
        )
        target = canonical.to_pixels(marked.center)
        landed = project(truth, (Point(x=target.x, y=target.y),))[0]
        assert composited[round(landed.y), round(landed.x)] < INK_LEVEL

    def test_the_rest_of_the_page_is_untouched(self, template, tmp_path):
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        reference = load_reference_scan(path, template)
        layer_spec = sheet_spec_from_template(
            template,
            {"questions_0": {0: MarkPlan(labels=("B",))}},
            render=PageRender(
                width=reference.mark_layer_size[0],
                height=reference.mark_layer_size[1],
                dpi=0,
                derived_from="reference_scan",
            ),
        )
        composited = render_onto_reference(reference, render_mark_layer(layer_spec))
        changed = np.count_nonzero(composited != reference.image)
        assert 0 < changed < composited.size * 0.01

    def test_the_reference_itself_is_reusable(self, template, tmp_path):
        # One reference serves every sheet of a run, so compositing must not
        # accumulate marks into it.
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        reference = load_reference_scan(path, template)
        before = reference.image.copy()
        layer_spec = sheet_spec_from_template(
            template,
            {"questions_0": {0: MarkPlan(labels=("A",))}},
            render=PageRender(
                width=reference.mark_layer_size[0],
                height=reference.mark_layer_size[1],
                dpi=0,
                derived_from="reference_scan",
            ),
        )
        render_onto_reference(reference, render_mark_layer(layer_spec))
        render_onto_reference(reference, render_mark_layer(layer_spec))
        assert np.array_equal(reference.image, before)

    def test_a_layer_of_the_wrong_size_is_refused(self, template, tmp_path):
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        reference = load_reference_scan(path, template)
        wrong = sheet_spec_from_template(
            template,
            {},
            render=PageRender(width=100, height=140, dpi=0, derived_from="canonical"),
        )
        with pytest.raises(ValueError, match="expects"):
            render_onto_reference(reference, render_mark_layer(wrong))


class TestColourOfTheReference:
    def test_grayscale_by_default(self, template, tmp_path):
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        assert load_reference_scan(path, template).image.ndim == 2

    def test_colour_keeps_three_channels(self, template, tmp_path):
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        reference = load_reference_scan(path, template, color_mode=ColorMode.COLOR)
        assert reference.image.ndim == 3
        assert reference.describe()["channels"] == 3

    def test_black_and_white_is_captured_in_grey(self, template, tmp_path):
        # Quantisation happens after degradation, not when the reference is
        # read - see `ColorMode`.
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        reference = load_reference_scan(
            path, template, color_mode=ColorMode.BLACK_AND_WHITE
        )
        assert reference.image.ndim == 2
        assert len(np.unique(reference.image)) > 2


class TestWhatIsRecorded:
    def test_only_the_file_name_is_kept(self, template, tmp_path):
        # A dataset must be movable, and must not carry somebody's folder
        # layout into a manifest that may be shared.
        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        described = load_reference_scan(path, template).describe()
        assert described["name"] == "blank.png"
        assert str(tmp_path) not in repr(described)

    def test_the_description_is_json_safe(self, template, tmp_path):
        import json

        path, _truth = write_reference(template, tmp_path, DistortionSpec())
        described = load_reference_scan(path, template).describe()
        assert json.loads(json.dumps(described))["mark_layer_scale"] >= 1
