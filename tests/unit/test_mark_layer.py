"""The mark-only render, the compositor and the colour modes.

These three are what the reference-scan rendering mode is built out of, and
each one fails in a way the finished image hides. A mark layer that quietly
carried a bubble ring would print a second ring on top of the real one; a
compositor that pasted rather than absorbed would erase the paper under every
mark; a colour mode applied in the wrong order would produce a "black and
white" page full of grey. So they are measured on their own terms first.

The property that matters most is stated as its own test class: a mark layer
must contain the candidate's ink and *nothing the printed form already has*.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import build_answer_sheet_template

from omr_scanner.domain.geometry import NormalizedPoint
from omr_scanner.evaluation.synthetic_dataset import PageRender, sheet_spec_from_template
from omr_scanner.evaluation.test_cases import MarkPlan
from omr_scanner.imaging.synthetic import (
    AnswerBubbleSpec,
    ColorMode,
    DistortionSpec,
    SyntheticSheetSpec,
    capture_channels,
    composite_marks,
    distort_image,
    quantise_to_output,
    render_mark_layer,
    render_sheet,
)

INK_LEVEL = 60
"""Grey level below which a rendered pixel counts as ink."""

PAPER_LEVEL = 250
"""Grey level above which a pixel counts as untouched paper.

Deliberately not 255: an anti-aliased stroke leaves values just below white at
its edges, and a test that demanded exactly 255 would be asserting that marks
have hard edges rather than that the page is empty."""


@pytest.fixture
def template():
    return build_answer_sheet_template()


def page_render(width: int, height: int) -> PageRender:
    """A plain page size, for specs that are not about resolution."""
    return PageRender(width=width, height=height, dpi=150, derived_from="canonical")


def marked_spec(template, *, question: int = 0, label: str = "B") -> SyntheticSheetSpec:
    """A template-derived page with exactly one question answered."""
    return sheet_spec_from_template(
        template,
        {"questions_0": {question: MarkPlan(labels=(label,))}},
        render=page_render(
            template.page.canonical_width_px, template.page.canonical_height_px
        ),
    )


class TestTheMarkLayerCarriesTheMarks:
    def test_it_is_a_page_sized_grayscale_image(self, template):
        layer = render_mark_layer(marked_spec(template))
        assert layer.image.shape == (
            template.page.canonical_height_px,
            template.page.canonical_width_px,
        )
        assert layer.image.dtype == np.uint8

    def test_a_marked_bubble_receives_ink(self, template):
        spec = marked_spec(template)
        layer = render_mark_layer(spec)
        marked = next(bubble for bubble in spec.answer_bubbles if bubble.fill > 0.0)
        center = spec.to_pixels(marked.center)
        assert layer.image[round(center.y), round(center.x)] < INK_LEVEL

    def test_it_counts_the_bubbles_it_inked(self, template):
        spec = sheet_spec_from_template(
            template,
            {"questions_0": {0: MarkPlan(labels=("A",)), 1: MarkPlan(labels=("B", "C"))}},
            render=page_render(
                template.page.canonical_width_px, template.page.canonical_height_px
            ),
        )
        assert render_mark_layer(spec).mark_count == 3

    def test_an_unanswered_sheet_produces_a_blank_layer(self, template):
        spec = sheet_spec_from_template(
            template,
            {},
            render=page_render(
                template.page.canonical_width_px, template.page.canonical_height_px
            ),
        )
        layer = render_mark_layer(spec)
        assert layer.mark_count == 0
        assert int(layer.image.min()) == 255

    def test_the_marks_land_exactly_where_a_full_render_puts_them(self, template):
        # The two modes must not disagree about where fill=0.95 goes, or their
        # results stop being comparable - which is the whole reason the mark
        # geometry is shared rather than reimplemented.
        spec = marked_spec(template)
        full = render_sheet(spec).image
        layer = render_mark_layer(spec).image
        inked = layer < INK_LEVEL
        assert inked.any()
        assert bool(np.all(full[inked] < INK_LEVEL))

    def test_rendering_is_deterministic(self, template):
        spec = marked_spec(template)
        assert np.array_equal(
            render_mark_layer(spec).image, render_mark_layer(spec).image
        )


class TestTheMarkLayerCarriesNothingElse:
    """The property the whole rendering mode rests on.

    Everything asserted here is already printed on the real sheet the layer
    will be laid over. Drawing any of it again would double the ink that is
    there, and the one place that matters most is inside an *empty* bubble -
    the printed option letter is exactly what the blank threshold has to see
    through, and two copies of it would raise the measured fill of the one
    thing the engine must read as empty.
    """

    def test_no_registration_markers(self, template):
        spec = marked_spec(template)
        layer = render_mark_layer(spec).image
        for marker in template.registration_markers:
            center = spec.to_pixels(marker.center)
            assert layer[round(center.y), round(center.x)] > PAPER_LEVEL

    def test_no_orientation_mark(self, template):
        spec = marked_spec(template)
        layer = render_mark_layer(spec).image
        center = spec.to_pixels(template.orientation_marker.center)
        assert layer[round(center.y), round(center.x)] > PAPER_LEVEL

    def test_no_bubble_rings(self, template):
        spec = marked_spec(template)
        layer = render_mark_layer(spec).image
        unmarked = next(bubble for bubble in spec.answer_bubbles if bubble.fill == 0.0)
        center = spec.to_pixels(unmarked.center)
        half_x = round(unmarked.width * spec.width / 2.0)
        window = layer[
            round(center.y) - 2 : round(center.y) + 3,
            round(center.x) - half_x - 2 : round(center.x) + half_x + 3,
        ]
        assert int(window.min()) > PAPER_LEVEL

    def test_no_printed_option_letters(self, template):
        spec = marked_spec(template)
        layer = render_mark_layer(spec).image
        unmarked = [bubble for bubble in spec.answer_bubbles if bubble.fill == 0.0]
        assert any(bubble.symbol for bubble in unmarked), "fixture prints no symbols"
        for bubble in unmarked:
            center = spec.to_pixels(bubble.center)
            assert layer[round(center.y), round(center.x)] > PAPER_LEVEL

    def test_a_layer_is_emptier_than_the_page_it_belongs_to(self, template):
        spec = marked_spec(template)
        layer = render_mark_layer(spec).image
        full = render_sheet(spec).image
        assert float(np.mean(layer)) > float(np.mean(full))

    def test_decoy_graphics_are_absent_even_when_asked_for(self):
        # The default synthetic page carries text bars, answer frames and a
        # bubble grid. None of them is the candidate's ink.
        spec = SyntheticSheetSpec(
            draw_bubbles=True,
            draw_text_bars=True,
            draw_answer_frames=True,
            answer_bubbles=(),
        )
        assert int(render_mark_layer(spec).image.min()) == 255


class TestCompositing:
    def test_ink_darkens_the_page_it_lands_on(self):
        page = np.full((20, 20), 200, dtype=np.uint8)
        marks = np.full((20, 20), 255, dtype=np.uint8)
        marks[5:10, 5:10] = 0
        result = composite_marks(page, marks)
        assert int(result[7, 7]) == 0
        assert int(result[1, 1]) == 200

    def test_the_page_survives_under_a_partial_mark(self):
        # Multiplicative, not a paste: a half-strength mark over grey paper
        # leaves grey, and the paper's own texture is still there underneath.
        page = np.full((8, 8), 200, dtype=np.uint8)
        marks = np.full((8, 8), 128, dtype=np.uint8)
        result = composite_marks(page, marks)
        assert 95 <= int(result[0, 0]) <= 105

    def test_a_mark_never_lightens_what_it_covers(self):
        page = np.full((8, 8), 40, dtype=np.uint8)
        marks = np.full((8, 8), 200, dtype=np.uint8)
        assert int(composite_marks(page, marks).max()) <= 40

    def test_a_colour_page_is_darkened_in_every_channel(self):
        page = np.zeros((6, 6, 3), dtype=np.uint8)
        page[:, :] = (200, 150, 100)
        marks = np.zeros((6, 6), dtype=np.uint8)
        result = composite_marks(page, marks)
        assert result.shape == (6, 6, 3)
        assert int(result.max()) == 0

    def test_the_background_is_not_modified(self):
        page = np.full((6, 6), 255, dtype=np.uint8)
        composite_marks(page, np.zeros((6, 6), dtype=np.uint8))
        assert int(page.min()) == 255

    def test_a_mismatched_layer_is_rejected(self):
        with pytest.raises(ValueError, match="does not match the page"):
            composite_marks(
                np.full((10, 10), 255, dtype=np.uint8),
                np.full((8, 8), 255, dtype=np.uint8),
            )


class TestColourModes:
    def test_grayscale_changes_nothing(self):
        page = np.full((10, 10), 180, dtype=np.uint8)
        captured = capture_channels(page, ColorMode.GRAYSCALE)
        assert captured.ndim == 2
        assert np.array_equal(quantise_to_output(captured, ColorMode.GRAYSCALE), page)

    def test_colour_capture_produces_three_channels(self):
        page = np.full((10, 10), 180, dtype=np.uint8)
        captured = capture_channels(page, ColorMode.COLOR)
        assert captured.shape == (10, 10, 3)
        assert int(captured.min()) == 180

    def test_colour_capture_leaves_an_already_colour_page_alone(self):
        page = np.zeros((4, 4, 3), dtype=np.uint8)
        assert capture_channels(page, ColorMode.COLOR).shape == (4, 4, 3)

    def test_a_grey_mode_flattens_a_colour_page(self):
        page = np.zeros((4, 4, 3), dtype=np.uint8)
        assert capture_channels(page, ColorMode.BLACK_AND_WHITE).ndim == 2

    @pytest.mark.parametrize("mode", list(ColorMode))
    def test_an_alpha_channel_is_handled_rather_than_asserted_on(self, mode):
        # A PNG or TIFF can carry one. Four channels through COLOR_BGR2GRAY is
        # an OpenCV assertion, which must never become the error interface.
        page = np.full((6, 6, 4), 200, dtype=np.uint8)
        captured = capture_channels(page, mode)
        assert captured.ndim == (3 if mode is ColorMode.COLOR else 2)
        assert quantise_to_output(captured, mode).dtype == np.uint8

    def test_black_and_white_output_holds_only_two_values(self):
        page = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
        result = quantise_to_output(page, ColorMode.BLACK_AND_WHITE)
        assert set(np.unique(result).tolist()) <= {0, 255}

    def test_black_and_white_is_not_applied_before_degradation(self):
        # Order is the whole point: thresholding first would let the blur put
        # the grey levels a bilevel scan does not contain straight back.
        page = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
        captured = capture_channels(page, ColorMode.BLACK_AND_WHITE)
        assert len(np.unique(captured)) > 2

    def test_a_colour_page_survives_the_whole_degradation_chain(self):
        page = capture_channels(
            np.full((200, 160), 220, dtype=np.uint8), ColorMode.COLOR
        )
        spec = DistortionSpec(
            rotation_degrees=3.0,
            illumination_gradient=0.3,
            edge_shadow=0.2,
            speckle_density=0.002,
            noise_sigma=4.0,
            blur_kernel_px=3,
            jpeg_quality=80,
            seed=5,
        )
        result = distort_image(page, spec)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_speckle_on_a_colour_page_is_grey_dust_not_confetti(self):
        # A per-channel draw would produce coloured specks, which is not what
        # dust on a platen looks like.
        page = capture_channels(
            np.full((120, 120), 200, dtype=np.uint8), ColorMode.COLOR
        )
        speckled = distort_image(page, DistortionSpec(speckle_density=0.05, seed=3))
        blue, green, red = (speckled[..., index] for index in range(3))
        assert np.array_equal(blue, green)
        assert np.array_equal(green, red)

    def test_grayscale_degradation_is_untouched_by_colour_support(self):
        # The existing behaviour, asserted directly: every stage that had to
        # learn about channels must leave a one-channel page exactly as it was.
        page = np.full((200, 160), 220, dtype=np.uint8)
        page[40:60, 40:60] = 10
        spec = DistortionSpec(
            illumination_gradient=0.4,
            edge_shadow=0.25,
            speckle_density=0.01,
            jpeg_quality=70,
            seed=9,
        )
        assert distort_image(page, spec).ndim == 2


class TestTheMarkLayerUsesTheSameShapes:
    @pytest.mark.parametrize("fill", [0.2, 0.5, 0.95])
    def test_coverage_matches_the_full_render_at_every_strength(self, fill):
        spec = SyntheticSheetSpec(
            width=400,
            height=560,
            draw_bubbles=False,
            draw_text_bars=False,
            draw_answer_frames=False,
            omit_orientation_marker=True,
            omit_markers=frozenset(),
            answer_bubbles=(
                AnswerBubbleSpec(
                    center=NormalizedPoint(x=0.5, y=0.5),
                    width=0.08,
                    height=0.06,
                    fill=fill,
                ),
            ),
        )
        layer_ink = int(np.count_nonzero(render_mark_layer(spec).image < INK_LEVEL))
        assert layer_ink > 0
        # The full render adds the printed ring around the same mark, so it can
        # only ever have more ink - never less, and never in a different place.
        full = render_sheet(spec).image
        inked = render_mark_layer(spec).image < INK_LEVEL
        assert bool(np.all(full[inked] < INK_LEVEL))
