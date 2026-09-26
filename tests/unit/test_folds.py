"""Folding a page corner: the geometry, and what it does to the pixels.

Why these are measured rather than looked at:
    A fold is easy to make *look* right and easy to get wrong in ways a
    screenshot hides - a crease a few pixels off, a flap reflected about the
    wrong axis, a shadow that quietly darkens the whole page. Each of those
    would produce a dataset that tests something other than what it claims.

    So the geometry is checked against arithmetic that does not go through the
    renderer (a reflection is an involution; a crease is fixed by it), and the
    rasterisation is checked by what it changed and, just as importantly, by
    what it left alone.

The property most of this file exists for:
    **A fold is local.** Everything outside the folded corner's own
    neighbourhood must be untouched, because that is the whole difference
    between a fold and a distortion. A global homography can be undone; a fold
    cannot, and a "fold" that moved the middle of the page would be neither.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from omr_scanner.imaging.folds import (
    CORNER_ORDER,
    MAX_DEPTH,
    MIN_DEPTH,
    SEVERITY_DEPTH_RANGES,
    FoldCorner,
    FoldSeverity,
    FoldSpec,
    PagePlacement,
    apply_corner_fold,
    apply_corner_folds,
    crease,
    fold_region,
    fold_triangle,
    overlap_fraction,
    project_points,
    reflection_matrix,
    severity_for_depth,
)
from omr_scanner.imaging.models import Point

WIDTH, HEIGHT = 1240, 1754
"""A4 at 150 dpi: the canonical page the rest of the suite uses."""

INK = 60
"""Grey level below which a pixel counts as ink."""


@pytest.fixture
def page() -> np.ndarray:
    """A page with structure everywhere, so any change shows up somewhere.

    A blank page would make "the fold changed nothing here" trivially true and
    "the fold changed something there" nearly unprovable.
    """
    image = np.full((HEIGHT, WIDTH), 250, dtype=np.uint8)
    image[::40, :] = 90
    image[:, ::40] = 90
    return image


def marker_at(x: float, y: float, width: float, height: float) -> tuple[Point, ...]:
    """A centred rectangle in page pixels, clockwise from the top-left."""
    half_x, half_y = width / 2.0, height / 2.0
    return (
        Point(x=x - half_x, y=y - half_y),
        Point(x=x + half_x, y=y - half_y),
        Point(x=x + half_x, y=y + half_y),
        Point(x=x - half_x, y=y + half_y),
    )


def moderate(corner: FoldCorner, depth_x: float = 0.05, depth_y: float = 0.035) -> FoldSpec:
    """A mid-sized fold with deliberately unequal depths."""
    return FoldSpec(
        corner=corner,
        severity=FoldSeverity.MODERATE,
        depth_x=depth_x,
        depth_y=depth_y,
    )


# ----------------------------------------------------------------------
# A. Geometry, at every corner
# ----------------------------------------------------------------------
class TestTheFoldGeometry:
    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_the_triangle_starts_at_its_own_corner(self, corner):
        origin, along_x, along_y = fold_triangle(moderate(corner), WIDTH, HEIGHT)
        expected = corner.origin(WIDTH, HEIGHT)
        assert (origin.x, origin.y) == (expected.x, expected.y)
        # The two intercepts leave along one edge each, so each shares exactly
        # one coordinate with the corner.
        assert along_x.y == origin.y
        assert along_y.x == origin.x

    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_the_intercepts_move_into_the_page(self, corner):
        origin, along_x, along_y = fold_triangle(moderate(corner), WIDTH, HEIGHT)
        assert 0.0 <= along_x.x <= WIDTH
        assert 0.0 <= along_y.y <= HEIGHT
        # Towards the middle, never off the sheet.
        assert abs(along_x.x - WIDTH / 2) < abs(origin.x - WIDTH / 2)
        assert abs(along_y.y - HEIGHT / 2) < abs(origin.y - HEIGHT / 2)

    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_the_depths_are_fractions_of_the_page(self, corner):
        spec = moderate(corner, depth_x=0.05, depth_y=0.035)
        origin, along_x, along_y = fold_triangle(spec, WIDTH, HEIGHT)
        assert abs(along_x.x - origin.x) == pytest.approx(0.05 * WIDTH)
        assert abs(along_y.y - origin.y) == pytest.approx(0.035 * HEIGHT)

    def test_the_crease_joins_the_two_intercepts(self):
        spec = moderate(FoldCorner.TOP_LEFT)
        _origin, along_x, along_y = fold_triangle(spec, WIDTH, HEIGHT)
        first, second = crease(spec, WIDTH, HEIGHT)
        assert (first.x, first.y) == (along_x.x, along_x.y)
        assert (second.x, second.y) == (along_y.x, along_y.y)

    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_the_region_is_the_kite_not_just_the_triangle(self, corner):
        """Both halves, which is the thing most easily got wrong.

        The corner triangle is vacated; its mirror image is covered by the
        flap. Measuring marker overlap against the triangle alone would
        understate every fold by about half.
        """
        spec = moderate(corner)
        region = fold_region(spec, WIDTH, HEIGHT)
        assert len(region) == 4
        origin, along_x, along_y = fold_triangle(spec, WIDTH, HEIGHT)
        # The first, second and fourth points are the triangle; the third is
        # the corner's reflection, on the far side of the crease.
        assert (region[0].x, region[0].y) == (origin.x, origin.y)
        assert (region[1].x, region[1].y) == (along_x.x, along_x.y)
        assert (region[3].x, region[3].y) == (along_y.x, along_y.y)
        mirror = reflection_matrix(along_x, along_y)
        expected = project_points(mirror, (origin,))[0]
        assert region[2].x == pytest.approx(expected.x)
        assert region[2].y == pytest.approx(expected.y)

    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_the_flap_lands_inside_the_page(self, corner):
        # A flap that reached off the sheet would mean the fold was deeper
        # than the page, which `FoldSpec` already refuses - this checks the
        # arithmetic agrees.
        region = fold_region(moderate(corner), WIDTH, HEIGHT)
        for point in region:
            assert -1.0 <= point.x <= WIDTH + 1.0
            assert -1.0 <= point.y <= HEIGHT + 1.0


class TestReflection:
    def test_it_is_its_own_inverse(self):
        matrix = reflection_matrix(Point(x=100.0, y=0.0), Point(x=0.0, y=140.0))
        assert np.allclose(matrix @ matrix, np.eye(3))

    def test_it_leaves_the_crease_alone(self):
        first, second = Point(x=100.0, y=0.0), Point(x=0.0, y=140.0)
        moved = project_points(reflection_matrix(first, second), (first, second))
        for before, after in zip((first, second), moved, strict=True):
            assert after.x == pytest.approx(before.x)
            assert after.y == pytest.approx(before.y)

    def test_a_degenerate_crease_is_refused(self):
        with pytest.raises(ValueError, match="two distinct endpoints"):
            reflection_matrix(Point(x=5.0, y=5.0), Point(x=5.0, y=5.0))


# ----------------------------------------------------------------------
# B. Severity
# ----------------------------------------------------------------------
class TestSeverity:
    def test_the_bands_are_contiguous_and_ascending(self):
        bounds = [SEVERITY_DEPTH_RANGES[severity] for severity in FoldSeverity]
        for (_low, high), (next_low, _next_high) in pairwise(bounds):
            assert high == pytest.approx(next_low)
        assert bounds[0][0] == MIN_DEPTH
        assert bounds[-1][1] == MAX_DEPTH

    @pytest.mark.parametrize("severity", list(FoldSeverity))
    def test_every_band_names_itself(self, severity):
        low, high = SEVERITY_DEPTH_RANGES[severity]
        middle = (low + high) / 2.0
        assert severity_for_depth(middle) is severity

    def test_a_depth_beyond_the_bands_is_still_named(self):
        # A fold solved to reach a particular marker may land outside the
        # nominal range, and an unclassified sheet would drop out of every
        # benchmark category.
        assert severity_for_depth(0.4) is FoldSeverity.SEVERE
        assert severity_for_depth(0.0001) is FoldSeverity.MICRO

    def test_deeper_severities_really_are_deeper(self, page):
        areas = []
        for severity in FoldSeverity:
            low, high = SEVERITY_DEPTH_RANGES[severity]
            depth = (low + high) / 2.0
            spec = FoldSpec(
                corner=FoldCorner.TOP_LEFT,
                severity=severity,
                depth_x=depth,
                depth_y=depth,
            )
            folded = apply_corner_fold(page, spec)
            areas.append(int(np.count_nonzero(folded != page)))
        assert areas == sorted(areas)


class TestTheSpecRefusesNonsense:
    @pytest.mark.parametrize("depth", [0.0, -0.01, 0.51, 1.0])
    def test_an_impossible_depth(self, depth):
        with pytest.raises(ValueError, match="depth_"):
            FoldSpec(
                corner=FoldCorner.TOP_LEFT,
                severity=FoldSeverity.MODERATE,
                depth_x=depth,
                depth_y=0.05,
            )

    def test_an_impossible_shading(self):
        with pytest.raises(ValueError, match="shadow"):
            FoldSpec(
                corner=FoldCorner.TOP_LEFT,
                severity=FoldSeverity.MODERATE,
                depth_x=0.05,
                depth_y=0.05,
                shadow=1.5,
            )


# ----------------------------------------------------------------------
# C. Locality - the property that makes this a fold and not a distortion
# ----------------------------------------------------------------------
class TestLocality:
    @pytest.mark.parametrize("corner", list(FoldCorner))
    @pytest.mark.parametrize("depth", [0.004, 0.02, 0.05, 0.11])
    def test_the_middle_of_the_page_is_untouched(self, page, corner, depth):
        spec = FoldSpec(
            corner=corner,
            severity=severity_for_depth(depth),
            depth_x=depth,
            depth_y=depth * 0.7,
        )
        folded = apply_corner_fold(page, spec)
        middle = (slice(HEIGHT // 4, 3 * HEIGHT // 4), slice(WIDTH // 4, 3 * WIDTH // 4))
        assert np.array_equal(page[middle], folded[middle])

    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_every_change_is_near_its_own_corner(self, page, corner):
        spec = moderate(corner, depth_x=0.06, depth_y=0.06)
        folded = apply_corner_fold(page, spec)
        changed = np.argwhere(folded != page)
        assert changed.size > 0
        origin = corner.origin(WIDTH, HEIGHT)
        # Generous, because the shadow reaches a little past the flap - but
        # far short of the page, which is the claim.
        reach = 0.20
        for y, x in changed[:: max(1, len(changed) // 500)]:
            assert abs(x - origin.x) <= reach * WIDTH
            assert abs(y - origin.y) <= reach * HEIGHT

    def test_the_opposite_corner_is_untouched(self, page):
        folded = apply_corner_fold(page, moderate(FoldCorner.TOP_LEFT, 0.11, 0.11))
        opposite = (slice(HEIGHT - 300, HEIGHT), slice(WIDTH - 300, WIDTH))
        assert np.array_equal(page[opposite], folded[opposite])

    def test_the_source_page_is_never_modified(self, page):
        before = page.copy()
        apply_corner_fold(page, moderate(FoldCorner.TOP_LEFT))
        assert np.array_equal(page, before)


# ----------------------------------------------------------------------
# D. Asymmetric depth
# ----------------------------------------------------------------------
class TestAsymmetricDepth:
    def test_the_two_depths_are_independent(self):
        spec = FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.MODERATE,
            depth_x=0.06,
            depth_y=0.02,
        )
        _origin, along_x, along_y = fold_triangle(spec, WIDTH, HEIGHT)
        assert along_x.x == pytest.approx(0.06 * WIDTH)
        assert along_y.y == pytest.approx(0.02 * HEIGHT)

    def test_a_lopsided_fold_is_not_a_right_isoceles_triangle(self):
        # The default shape if the two depths were ever collapsed into one.
        spec = FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.MODERATE,
            depth_x=0.06,
            depth_y=0.02,
        )
        _origin, along_x, along_y = fold_triangle(spec, WIDTH, HEIGHT)
        assert along_x.x != pytest.approx(along_y.y)

    def test_swapping_the_depths_gives_a_different_page(self, page):
        wide = apply_corner_fold(
            page,
            FoldSpec(
                corner=FoldCorner.TOP_LEFT,
                severity=FoldSeverity.MODERATE,
                depth_x=0.07,
                depth_y=0.02,
            ),
        )
        tall = apply_corner_fold(
            page,
            FoldSpec(
                corner=FoldCorner.TOP_LEFT,
                severity=FoldSeverity.MODERATE,
                depth_x=0.02,
                depth_y=0.07,
            ),
        )
        assert not np.array_equal(wide, tall)


# ----------------------------------------------------------------------
# E. Micro folds
# ----------------------------------------------------------------------
class TestMicroFolds:
    @pytest.mark.parametrize("corner", list(FoldCorner))
    def test_a_micro_fold_survives_rasterisation(self, page, corner):
        """Only a few pixels deep, and it must still be there.

        Anti-aliased masks exist for exactly this: a triangle three pixels on a
        side drawn with integer vertices rounds to nothing at all.
        """
        spec = FoldSpec(
            corner=corner,
            severity=FoldSeverity.MICRO,
            depth_x=MIN_DEPTH,
            depth_y=MIN_DEPTH,
        )
        folded = apply_corner_fold(page, spec)
        assert int(np.count_nonzero(folded != page)) > 0

    def test_a_micro_fold_stays_tiny(self, page):
        spec = FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.MICRO,
            depth_x=MIN_DEPTH,
            depth_y=MIN_DEPTH,
        )
        folded = apply_corner_fold(page, spec)
        changed = np.argwhere(folded != page)
        assert changed[:, 0].max() < 0.02 * HEIGHT
        assert changed[:, 1].max() < 0.02 * WIDTH

    def test_it_is_not_quietly_enlarged_to_look_better(self):
        spec = FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.MICRO,
            depth_x=0.004,
            depth_y=0.004,
        )
        _origin, along_x, _along_y = fold_triangle(spec, WIDTH, HEIGHT)
        assert along_x.x == pytest.approx(0.004 * WIDTH)


# ----------------------------------------------------------------------
# F. What the fold does to the page's content
# ----------------------------------------------------------------------
class TestWhatIsTakenOutOfView:
    def build(self) -> np.ndarray:
        """A page carrying printed artwork *and* a candidate's mark in one corner."""
        image = np.full((HEIGHT, WIDTH), 250, dtype=np.uint8)
        # Printed: a registration-marker-shaped block.
        image[43:80, 43:81] = 0
        # The candidate's: a filled bubble, a little further in.
        image[95:115, 100:124] = 0
        return image

    def fold(self) -> FoldSpec:
        return FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.SEVERE,
            depth_x=0.12,
            depth_y=0.10,
        )

    def test_printed_artwork_inside_the_fold_stops_being_there(self):
        page = self.build()
        folded = apply_corner_fold(page, self.fold())
        assert page[60, 60] < INK
        assert folded[60, 60] > INK

    def test_a_candidate_mark_inside_the_fold_goes_with_it(self):
        """The point of folding the *composed* sheet rather than a layer.

        Paper does not fold selectively. A mark inside the fold has to leave
        its position exactly as the printing around it does - if it survived
        where the printing vanished, the model would be describing something
        that cannot happen.
        """
        page = self.build()
        folded = apply_corner_fold(page, self.fold())
        assert page[105, 112] < INK
        assert folded[105, 112] > INK

    def test_content_outside_the_fold_is_untouched(self):
        page = self.build()
        page[900:930, 600:640] = 0
        folded = apply_corner_fold(page, self.fold())
        assert np.array_equal(page[880:950, 580:660], folded[880:950, 580:660])

    def test_the_fold_leaves_paper_rather_than_a_hole(self):
        # Not a white triangle: the gap is backing, the flap is shaded paper,
        # and the crease is a line. So the folded corner has *structure*.
        page = self.build()
        folded = apply_corner_fold(page, self.fold())
        corner = folded[0:200, 0:200]
        assert len(np.unique(corner)) > 3

    def test_the_crease_is_darker_than_the_paper_it_crosses(self):
        page = np.full((HEIGHT, WIDTH), 250, dtype=np.uint8)
        spec = self.fold()
        folded = apply_corner_fold(page, spec)
        first, second = crease(spec, WIDTH, HEIGHT)
        midpoint = (
            round((first.y + second.y) / 2.0),
            round((first.x + second.x) / 2.0),
        )
        assert folded[midpoint] < 250


# ----------------------------------------------------------------------
# G. Determinism, colour and several corners
# ----------------------------------------------------------------------
class TestDeterminism:
    def test_the_same_fold_twice_is_the_same_pixels(self, page):
        spec = moderate(FoldCorner.BOTTOM_RIGHT, 0.06, 0.04)
        assert np.array_equal(
            apply_corner_fold(page, spec), apply_corner_fold(page, spec)
        )

    def test_different_depths_differ(self, page):
        assert not np.array_equal(
            apply_corner_fold(page, moderate(FoldCorner.TOP_LEFT, 0.05, 0.04)),
            apply_corner_fold(page, moderate(FoldCorner.TOP_LEFT, 0.06, 0.04)),
        )


class TestColour:
    def test_a_colour_page_keeps_its_channels(self, page):
        colour = np.dstack([page, page, page])
        folded = apply_corner_fold(colour, moderate(FoldCorner.TOP_LEFT))
        assert folded.shape == colour.shape
        assert folded.dtype == np.uint8

    def test_the_fold_is_neutral_rather_than_tinted(self, page):
        # Folded paper is not a colour effect; every channel must be treated
        # the same or the flap comes out coloured.
        colour = np.dstack([page, page, page])
        folded = apply_corner_fold(colour, moderate(FoldCorner.TOP_LEFT, 0.08, 0.06))
        blue, green, red = (folded[..., index] for index in range(3))
        assert np.array_equal(blue, green)
        assert np.array_equal(green, red)

    def test_a_colour_fold_changes_the_same_pixels_as_a_grey_one(self, page):
        spec = moderate(FoldCorner.TOP_LEFT, 0.08, 0.06)
        grey_changed = apply_corner_fold(page, spec) != page
        colour = np.dstack([page, page, page])
        colour_changed = (apply_corner_fold(colour, spec) != colour).any(axis=2)
        assert np.array_equal(grey_changed, colour_changed)


class TestSeveralCorners:
    def test_all_four_can_be_folded_at_once(self, page):
        specs = [moderate(corner, 0.05, 0.04) for corner in FoldCorner]
        folded = apply_corner_folds(page, specs)
        for corner in FoldCorner:
            origin = corner.origin(WIDTH, HEIGHT)
            probe = (
                round(min(max(origin.y, 5), HEIGHT - 6)),
                round(min(max(origin.x, 5), WIDTH - 6)),
            )
            assert folded[probe] != page[probe], corner

    def test_the_centre_survives_four_folds(self, page):
        specs = [moderate(corner, 0.11, 0.11) for corner in FoldCorner]
        folded = apply_corner_folds(page, specs)
        middle = (slice(HEIGHT // 3, 2 * HEIGHT // 3), slice(WIDTH // 3, 2 * WIDTH // 3))
        assert np.array_equal(page[middle], folded[middle])

    def test_the_order_they_are_listed_in_does_not_matter(self, page):
        specs = [moderate(corner, 0.05, 0.04) for corner in FoldCorner]
        assert np.array_equal(
            apply_corner_folds(page, specs),
            apply_corner_folds(page, list(reversed(specs))),
        )

    def test_no_folds_returns_the_page_itself(self, page):
        assert apply_corner_folds(page, []) is page


# ----------------------------------------------------------------------
# H. Overlap arithmetic
# ----------------------------------------------------------------------
class TestOverlapFraction:
    def test_a_marker_well_inside_the_fold_is_fully_covered(self):
        spec = FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.SEVERE,
            depth_x=0.12,
            depth_y=0.12,
        )
        marker = marker_at(0.05 * WIDTH, 0.035 * HEIGHT, 0.03 * WIDTH, 0.021 * HEIGHT)
        assert overlap_fraction(fold_region(spec, WIDTH, HEIGHT), marker) == pytest.approx(
            1.0, abs=1e-3
        )

    def test_a_marker_far_from_the_fold_is_untouched(self):
        spec = moderate(FoldCorner.TOP_LEFT)
        marker = marker_at(0.95 * WIDTH, 0.965 * HEIGHT, 0.03 * WIDTH, 0.021 * HEIGHT)
        assert overlap_fraction(fold_region(spec, WIDTH, HEIGHT), marker) == 0.0

    def test_overlap_grows_with_depth(self):
        marker = marker_at(0.05 * WIDTH, 0.035 * HEIGHT, 0.03 * WIDTH, 0.021 * HEIGHT)
        fractions = [
            overlap_fraction(
                fold_region(
                    FoldSpec(
                        corner=FoldCorner.TOP_LEFT,
                        severity=severity_for_depth(depth),
                        depth_x=depth,
                        depth_y=depth,
                    ),
                    WIDTH,
                    HEIGHT,
                ),
                marker,
            )
            for depth in (0.01, 0.03, 0.045, 0.06, 0.09)
        ]
        assert fractions == sorted(fractions)
        assert fractions[0] == 0.0
        assert fractions[-1] == pytest.approx(1.0, abs=1e-3)

    def test_it_is_a_fraction_of_the_marker_not_of_the_fold(self):
        # A large fold covering a small marker completely must report 1.0, not
        # the small share of itself the marker occupies.
        spec = FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.SEVERE,
            depth_x=0.12,
            depth_y=0.12,
        )
        tiny = marker_at(0.02 * WIDTH, 0.01 * HEIGHT, 0.004 * WIDTH, 0.004 * HEIGHT)
        assert overlap_fraction(fold_region(spec, WIDTH, HEIGHT), tiny) == pytest.approx(
            1.0, abs=1e-3
        )

    def test_a_degenerate_shape_overlaps_nothing(self):
        spec = moderate(FoldCorner.TOP_LEFT)
        assert overlap_fraction(fold_region(spec, WIDTH, HEIGHT), ()) == 0.0


# ----------------------------------------------------------------------
# I. Page placement - the coordinate-space seam
# ----------------------------------------------------------------------
class TestPagePlacement:
    def test_the_image_can_be_the_page(self, page):
        placement = PagePlacement.of_image(page)
        assert placement.width == WIDTH
        assert placement.height == HEIGHT
        assert np.allclose(placement.matrix, np.eye(3))

    def test_a_page_can_be_described_in_its_own_units(self):
        image = np.zeros((3508, 2480), dtype=np.uint8)
        placement = PagePlacement.scaled_to(image, width=WIDTH, height=HEIGHT)
        assert placement.width == WIDTH
        corner = project_points(placement.matrix, (Point(x=float(WIDTH), y=float(HEIGHT)),))[0]
        assert corner.x == pytest.approx(2480.0)
        assert corner.y == pytest.approx(3508.0)

    def test_the_two_axes_scale_independently(self):
        # A single scalar would skew every fold on a page whose declared
        # canonical size and physical aspect ratio disagree.
        image = np.zeros((1000, 2000), dtype=np.uint8)
        placement = PagePlacement.scaled_to(image, width=100.0, height=100.0)
        mapped = project_points(placement.matrix, (Point(x=100.0, y=100.0),))[0]
        assert mapped.x == pytest.approx(2000.0)
        assert mapped.y == pytest.approx(1000.0)

    def test_a_page_with_no_size_is_refused(self):
        with pytest.raises(ValueError, match="positive dimensions"):
            PagePlacement(width=0.0, height=10.0)

    def test_a_fold_follows_the_page_into_a_larger_image(self, page):
        """The reference-scan case: the page is somewhere else, and rotated.

        The fold must land on the *paper's* corner, not the image's, and the
        rest of the image must be untouched - including the platen area the
        page does not cover.
        """
        import cv2

        angle = np.deg2rad(6.0)
        transform = np.array(
            [
                [np.cos(angle) * 1.9, -np.sin(angle) * 1.9, 300.0],
                [np.sin(angle) * 2.1, np.cos(angle) * 2.1, 220.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scan = np.asarray(
            cv2.warpPerspective(page, transform, (3400, 4400), borderValue=255),
            dtype=np.uint8,
        )
        placement = PagePlacement(
            width=float(WIDTH), height=float(HEIGHT), to_image=transform
        )
        spec = FoldSpec(
            corner=FoldCorner.TOP_LEFT,
            severity=FoldSeverity.SEVERE,
            depth_x=0.10,
            depth_y=0.08,
        )
        folded = apply_corner_fold(scan, spec, placement)

        page_corner = project_points(transform, (Point(x=0.0, y=0.0),))[0]
        changed = np.argwhere(folded != scan)
        assert changed.size > 0
        assert abs(changed[:, 1].min() - page_corner.x) < 400
        assert abs(changed[:, 0].min() - page_corner.y) < 400
        assert np.array_equal(scan[2000:2400, 1500:1900], folded[2000:2400, 1500:1900])

    def test_a_fold_entirely_off_the_image_changes_nothing(self, page):
        # A reference scan cropped past the page corner. Drawing something
        # anyway would put a fold where there is no paper.
        placement = PagePlacement(
            width=float(WIDTH),
            height=float(HEIGHT),
            to_image=np.array(
                [[1.0, 0.0, -5000.0], [0.0, 1.0, -5000.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )
        folded = apply_corner_fold(page, moderate(FoldCorner.TOP_LEFT), placement)
        assert np.array_equal(folded, page)


def test_the_corner_order_matches_the_rest_of_the_package():
    """Clockwise from the top-left, like every other four-corner sequence."""
    assert CORNER_ORDER == (
        FoldCorner.TOP_LEFT,
        FoldCorner.TOP_RIGHT,
        FoldCorner.BOTTOM_RIGHT,
        FoldCorner.BOTTOM_LEFT,
    )
