"""Physically folded page corners, as a local deformation of a composed sheet.

Purpose:
    Reproduce the single most common physical accident an examination sheet
    suffers between the candidate's desk and the scanner: a corner folded over.
    It is worth reproducing because of what it does to *registration* - a fold
    large enough to reach a corner marker takes that marker out of view, and
    the engine's response to that is the behaviour worth measuring.

Why this is not a distortion:
    :class:`~omr_scanner.imaging.synthetic.DistortionSpec` is projective and
    photometric: every field of it describes something a homography or an
    exposure curve can express, which means a homography can undo it. A folded
    corner is neither. The paper has physically moved, part of the page is no
    longer in the image at all, and the rest of the sheet is untouched. No
    global transform describes that, and using one would be a different defect
    wearing a fold's name.

What a fold actually does to a sheet, which is what this models:
    Pick a corner ``P`` and two intercepts, ``A`` along one edge and ``B``
    along the other. The triangle ``PAB`` rotates about the crease ``AB`` and
    lands face-down on the page. Two regions change and nothing else does:

    * the triangle ``PAB`` is now **empty** - the paper that was there has gone
      somewhere else, and the scanner sees its backing through the gap;
    * the triangle's mirror image ``P'AB`` on the other side of the crease is
      **covered** by the back of the folded flap, which is blank paper that the
      printing ghosts faintly through.

    Together those two triangles form a kite ``P A P' B`` - see
    :func:`fold_region`. Everything inside it has stopped being readable;
    everything outside it is exactly as it was. That kite is what marker
    overlap is measured against, and it is why measuring against the corner
    triangle alone would understate a fold: the flap covers as much page as the
    gap exposes.

Coordinate spaces, which this module is deliberately strict about:
    A fold is defined in **page coordinates** - the flat rectangle the paper
    is, ``width`` x ``height``, with depths given as fractions of those. That
    is where the physics lives and where marker overlap is meaningful, because
    a fraction of a marker's area is a property of the paper.

    Where those page coordinates *land in the image* is a separate question,
    answered by :class:`PagePlacement`. For a synthetic page the two are the
    same and the transform is the identity. For a real scanned sheet the page
    is somewhere inside a larger image, rotated, possibly in perspective, and
    the transform is the registration homography Phase 1 produced. Reflection
    about the crease is an affine map ``M`` in page space; in image space it is
    ``H M H^-1``, which is still a homography, so the same code draws both
    without approximating anything - no single scalar scale, no assumed square
    pixels.

What does NOT belong here:
    * Templates, markers, fields or answers. This module folds a rectangle and
      reports which part of it went out of view; deciding that a particular
      part of the page was a registration marker is the caller's job.
    * Choosing *which* sheets to fold, or how hard. That is a dataset policy
      (:mod:`omr_scanner.evaluation.fold_plans`).

Scope, stated honestly:
    A lightweight raster model, not a simulation. There is no 3-D paper, no
    bending stiffness and no light transport. What it reproduces is the part
    that changes a recognition outcome - where the paper is, what is hidden,
    and a crease and shadow plausible enough that a detector meets an edge
    rather than a clean rectangle of white.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

import cv2
import numpy as np

from omr_scanner.imaging.geometry import polygon_area
from omr_scanner.imaging.models import Point

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class FoldSeverity(StrEnum):
    """How deep a fold is, as a named class rather than a number.

    The bands below are *guidance*, expressed as a fraction of the page's own
    dimensions so that a "small" fold is the same physical thing at 150 and at
    600 dpi. A fold's truth is its two depths; this names the band they were
    drawn from, which is what a benchmark groups by.
    """

    MICRO = "micro"
    """A corner turned over by a few millimetres. Often affects no marker at
    all, and at ordinary scan resolution is only a few pixels deep - which is
    the point: it must survive rasterisation rather than be rounded away."""

    SMALL = "small"
    """A visible dog-ear. Reaches a marker on some templates and not others."""

    MODERATE = "moderate"
    """Large enough to cover part of a corner marker on a typical sheet."""

    SEVERE = "severe"
    """A corner folded well into the page. Usually takes a registration marker
    with it, and the engine is expected to refuse the sheet rather than guess
    where the missing corner was."""


SEVERITY_DEPTH_RANGES: Mapping[FoldSeverity, tuple[float, float]] = MappingProxyType(
    {
        FoldSeverity.MICRO: (0.003, 0.015),
        FoldSeverity.SMALL: (0.015, 0.030),
        FoldSeverity.MODERATE: (0.030, 0.070),
        FoldSeverity.SEVERE: (0.070, 0.120),
    }
)
"""Depth band per severity, as a fraction of the page dimension.

Contiguous on purpose, so :func:`severity_for_depth` can name any depth and a
fold solved for a particular marker overlap still reports an honest class
rather than the class that was asked for."""

MIN_DEPTH = SEVERITY_DEPTH_RANGES[FoldSeverity.MICRO][0]
MAX_DEPTH = SEVERITY_DEPTH_RANGES[FoldSeverity.SEVERE][1]
"""The whole permitted range, end to end."""


def severity_for_depth(depth: float) -> FoldSeverity:
    """Return the severity band ``depth`` falls in.

    Clamped at both ends rather than raising: a fold solved to reach a
    particular marker may land just outside the nominal range, and refusing to
    name it would leave the dataset with an unclassified sheet.
    """
    for severity, (_low, high) in SEVERITY_DEPTH_RANGES.items():
        if depth <= high:
            return severity
    return FoldSeverity.SEVERE


class FoldCorner(StrEnum):
    """Which corner of the page is folded.

    Named for the page, not for the image: in the reference-scan mode the
    page's top-left corner may be anywhere in the scan, and may not be the
    top-left of it at all.
    """

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"

    def origin(self, width: float, height: float) -> Point:
        """The page corner itself, in page coordinates."""
        x = 0.0 if self in (FoldCorner.TOP_LEFT, FoldCorner.BOTTOM_LEFT) else width
        y = 0.0 if self in (FoldCorner.TOP_LEFT, FoldCorner.TOP_RIGHT) else height
        return Point(x=x, y=y)

    @property
    def direction(self) -> tuple[float, float]:
        """Unit steps *into* the page along each axis from this corner."""
        dx = 1.0 if self in (FoldCorner.TOP_LEFT, FoldCorner.BOTTOM_LEFT) else -1.0
        dy = 1.0 if self in (FoldCorner.TOP_LEFT, FoldCorner.TOP_RIGHT) else -1.0
        return dx, dy


CORNER_ORDER: tuple[FoldCorner, ...] = (
    FoldCorner.TOP_LEFT,
    FoldCorner.TOP_RIGHT,
    FoldCorner.BOTTOM_RIGHT,
    FoldCorner.BOTTOM_LEFT,
)
"""Clockwise, matching
:data:`~omr_scanner.imaging.models.CANONICAL_CORNER_ORDER` so the two can be
read side by side without translating."""


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One folded corner, fully determined.

    Carries no randomness: the same specification renders the same pixels every
    time, on any machine and in any worker process. Choosing the numbers is a
    planning decision and lives in
    :mod:`omr_scanner.evaluation.fold_plans`.

    Attributes:
        corner: Which corner went over.
        severity: The band the depths were drawn from. Recorded rather than
            derived, so a fold solved to hit a particular marker overlap still
            says which class of defect it was meant to be; the depths below
            remain the authoritative description.
        depth_x: How far along the horizontal edge the crease starts, as a
            fraction of page **width**.
        depth_y: The same along the vertical edge, as a fraction of page
            **height**. Independent of ``depth_x`` on purpose - a real fold is
            almost never a 45 degree triangle, and one that always was would
            make every fold in a dataset the same shape.
        flap_shade: How much darker the folded-back flap is than the paper
            around it, ``0`` for no shading.
        backing_shade: How much darker the gap left behind is than the paper.
        show_through: How strongly the printing on the flap's far side ghosts
            through it, ``0`` for opaque paper.
        shadow: Strength of the soft shadow the raised flap casts on the page.
        crease_darkness: How dark the crease line itself is.
    """

    corner: FoldCorner
    severity: FoldSeverity
    depth_x: float
    depth_y: float
    flap_shade: float = 0.07
    backing_shade: float = 0.12
    show_through: float = 0.18
    shadow: float = 0.35
    crease_darkness: float = 0.55

    def __post_init__(self) -> None:
        """Reject a fold that could not be drawn, or that is not local."""
        for name, value in (("depth_x", self.depth_x), ("depth_y", self.depth_y)):
            if not 0.0 < value <= 0.5:
                raise ValueError(
                    f"{name} must lie in (0, 0.5]; a fold deeper than half the "
                    f"page is not a folded corner. Got {value}"
                )
        for name, value in (
            ("flap_shade", self.flap_shade),
            ("backing_shade", self.backing_shade),
            ("show_through", self.show_through),
            ("shadow", self.shadow),
            ("crease_darkness", self.crease_darkness),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {value}")

    @property
    def depth(self) -> float:
        """The deeper of the two, which is what names the severity band."""
        return max(self.depth_x, self.depth_y)


@dataclass(frozen=True, slots=True)
class PagePlacement:
    """Where the page sits inside the image, and how large the page is.

    The whole of this module's coordinate discipline in one object. A fold is
    described on the page; this says how to get from there to pixels.

    Attributes:
        width: Page width, in page units.
        height: Page height, in page units.
        to_image: 3x3 homography mapping page coordinates to image pixels, or
            ``None`` when the image *is* the page - which is the
            template-rendered case, and is not the same thing as an identity
            matrix only in that it saves the arithmetic.
    """

    width: float
    height: float
    to_image: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Reject a placement that describes no page."""
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("A page must have positive dimensions")

    @classmethod
    def of_image(cls, image: NDArray[np.uint8]) -> PagePlacement:
        """The page *is* this image, and its own pixels are its page units."""
        return cls(width=float(image.shape[1]), height=float(image.shape[0]))

    @classmethod
    def scaled_to(
        cls, image: NDArray[np.uint8], *, width: float, height: float
    ) -> PagePlacement:
        """The page fills this image, but is described in its own units.

        For a sheet rendered from a template at some DPI: the image is the
        whole page, yet fold geometry and marker overlap are better expressed
        in the template's own canonical page size, so that the *same* fold
        means the same thing at 150 and 600 dpi and so that overlap fractions
        are comparable with the reference-scan mode's.

        The two axes scale independently, because a page's declared canonical
        size and its physical aspect ratio need not agree to the last pixel and
        collapsing them to one factor would quietly skew every fold.
        """
        scale = np.diag(
            np.array(
                [float(image.shape[1]) / width, float(image.shape[0]) / height, 1.0],
                dtype=np.float64,
            )
        )
        return cls(width=width, height=height, to_image=scale)

    @property
    def matrix(self) -> NDArray[np.float64]:
        """``to_image``, or the identity when the image is the page."""
        if self.to_image is None:
            return np.eye(3, dtype=np.float64)
        return np.asarray(self.to_image, dtype=np.float64)


# ----------------------------------------------------------------------
# Geometry, entirely in page coordinates
# ----------------------------------------------------------------------
def fold_triangle(spec: FoldSpec, width: float, height: float) -> tuple[Point, ...]:
    """Return ``(P, A, B)``: the corner and its two crease intercepts.

    ``A`` lies along the horizontal edge at ``depth_x`` of the page width;
    ``B`` along the vertical edge at ``depth_y`` of the page height. This is
    the paper that leaves the corner.
    """
    origin = spec.corner.origin(width, height)
    dx, dy = spec.corner.direction
    along_x = Point(x=origin.x + dx * spec.depth_x * width, y=origin.y)
    along_y = Point(x=origin.x, y=origin.y + dy * spec.depth_y * height)
    return (origin, along_x, along_y)


def crease(spec: FoldSpec, width: float, height: float) -> tuple[Point, Point]:
    """Return the crease's two endpoints, in page coordinates."""
    _origin, along_x, along_y = fold_triangle(spec, width, height)
    return along_x, along_y


def reflection_matrix(first: Point, second: Point) -> NDArray[np.float64]:
    """Return the 3x3 affine reflection about the line through two points.

    A point ``p`` reflects to ``p - 2((p - a) . n) n`` for a unit normal ``n``,
    which rearranges to ``(I - 2 n n^T) p + 2 (a . n) n`` - the matrix below.

    Raises:
        ValueError: The two points coincide, so they name no line.
    """
    dx = second.x - first.x
    dy = second.y - first.y
    length = float(np.hypot(dx, dy))
    if length == 0.0:
        raise ValueError("A crease needs two distinct endpoints")
    # Normal to the crease, unit length.
    nx, ny = -dy / length, dx / length
    offset = 2.0 * (first.x * nx + first.y * ny)
    return np.array(
        [
            [1.0 - 2.0 * nx * nx, -2.0 * nx * ny, offset * nx],
            [-2.0 * nx * ny, 1.0 - 2.0 * ny * ny, offset * ny],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fold_region(spec: FoldSpec, width: float, height: float) -> tuple[Point, ...]:
    """Return the kite ``P, A, P', B`` - everything the fold takes out of view.

    Both halves, and this is the part most easily got wrong. The corner
    triangle ``PAB`` is *vacated*: its paper has gone. Its mirror image
    ``P'AB`` is *covered*: the flap now lies on top of it. Content in either is
    no longer readable where it was, so marker overlap must be measured against
    the union - which, the two triangles sharing the crease as a diagonal, is a
    convex quadrilateral.

    Returned in order ``P, A, P', B`` so the outline traces the kite without
    crossing itself.
    """
    origin, along_x, along_y = fold_triangle(spec, width, height)
    mirror = reflection_matrix(along_x, along_y)
    reflected = mirror @ np.array([origin.x, origin.y, 1.0], dtype=np.float64)
    far = Point(x=float(reflected[0]), y=float(reflected[1]))
    return (origin, along_x, far, along_y)


def overlap_fraction(region: Sequence[Point], shape: Sequence[Point]) -> float:
    """Return how much of ``shape`` lies inside ``region``, in ``[0, 1]``.

    Both must be convex, which every polygon this module produces is: the fold
    kite by construction, and a marker because a printed registration mark is a
    rectangle (possibly a skewed one, once mapped into a scan).

    Deliberately a *fraction of the shape*, not of the region: the question
    being asked is "how much of this marker is gone", and answering with a
    fraction of the fold would give a large fold credit for covering a small
    marker completely.
    """
    if len(region) < 3 or len(shape) < 3:
        return 0.0
    area = polygon_area(shape)
    if area <= 0.0:
        return 0.0
    intersection, _points = cv2.intersectConvexConvex(
        _as_contour(region), _as_contour(shape)
    )
    return float(min(max(intersection / area, 0.0), 1.0))


def _as_contour(points: Sequence[Point]) -> NDArray[np.float32]:
    """Return points in the ``(N, 1, 2)`` float32 shape OpenCV expects."""
    return np.array(
        [[[point.x, point.y]] for point in points], dtype=np.float32
    )


def project_points(
    matrix: NDArray[np.float64], points: Sequence[Point]
) -> tuple[Point, ...]:
    """Map page points into image pixels through a homography."""
    if not points:
        return ()
    homogeneous = np.array([[p.x, p.y, 1.0] for p in points], dtype=np.float64)
    mapped = homogeneous @ matrix.T
    return tuple(
        Point(x=float(row[0] / row[2]), y=float(row[1] / row[2])) for row in mapped
    )


# ----------------------------------------------------------------------
# Rasterisation
# ----------------------------------------------------------------------
_SHADOW_SPREAD_RATIO = 0.35
"""How far the flap's shadow reaches beyond it, as a fraction of fold depth.

Bounded on purpose: a shadow that faded away over the whole page would make
"the fold is local" untrue, and locality is the property that distinguishes
this from a distortion."""

_MIN_ROI_MARGIN_PX = 3
"""Smallest working margin around a fold, in pixels. A micro fold's shadow and
anti-aliased crease still need somewhere to land."""

_PAPER_PERCENTILE = 90
"""Which percentile of the working area counts as "the paper".

Measured from the image rather than assumed to be 255, so that a fold on a real
scan takes the tone of the paper it is actually on - grey stock, a warm scanner,
an uneven exposure - instead of stamping a patch of pure white onto it."""


def apply_corner_fold(
    image: NDArray[np.uint8], spec: FoldSpec, placement: PagePlacement | None = None
) -> NDArray[np.uint8]:
    """Return ``image`` with one corner of its page folded over.

    Args:
        image: The **composed** sheet - printed artwork, registration marks and
            the candidate's own marks already on it. Read only. Folding a
            layer before composition would fold the paper without folding what
            is printed on it, which is not a thing that happens.
        spec: The fold.
        placement: Where the page sits in the image; defaults to "the image is
            the page".

    Returns:
        A new array, same shape and channel layout. Only pixels inside the
        fold's own neighbourhood differ - see :data:`_SHADOW_SPREAD_RATIO`.

    The composition, in the order the physics happens:

    1. the flap's content is the corner's content mirrored about the crease;
    2. the flap is blank paper with that content ghosting faintly through it;
    3. the gap the corner left shows the backing behind the sheet;
    4. the raised flap casts a soft shadow on the page around it;
    5. the crease itself is a hard line.
    """
    active = placement if placement is not None else PagePlacement.of_image(image)
    matrix = active.matrix

    page_region = fold_region(spec, active.width, active.height)
    image_region = project_points(matrix, page_region)

    height, width = image.shape[:2]
    margin = max(
        _MIN_ROI_MARGIN_PX,
        int(
            _SHADOW_SPREAD_RATIO
            * max(spec.depth_x * active.width, spec.depth_y * active.height)
        )
        + 2,
    )
    roi = _bounding_roi(image_region, width, height, margin)
    if roi is None:
        # The fold lies entirely outside the image - possible when a reference
        # scan is cropped past the page corner. Nothing to draw, and inventing
        # something would be worse than leaving the sheet alone.
        return image.copy()
    left, top, roi_width, roi_height = roi

    result = image.copy()
    window = result[top : top + roi_height, left : left + roi_width]
    paper = _paper_level(window)

    shift = np.array(
        [[1.0, 0.0, -float(left)], [0.0, 1.0, -float(top)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # Reflection about the crease is affine in page space; conjugating it by
    # the page-to-image transform gives the equivalent map in image space, and
    # reflection being its own inverse makes this the source lookup too.
    page_mirror = reflection_matrix(*crease(spec, active.width, active.height))
    image_mirror = matrix @ page_mirror @ np.linalg.inv(matrix)
    mirrored = np.asarray(
        cv2.warpPerspective(
            image,
            shift @ image_mirror,
            (roi_width, roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ),
        dtype=np.uint8,
    )

    # The kite's four points, local to the working window: corner, the two
    # crease intercepts either side of it, and the corner's mirror image.
    corner, along_x, mirrored_corner, along_y = (
        Point(x=point.x - left, y=point.y - top) for point in image_region
    )
    gap_mask = _polygon_mask((roi_height, roi_width), (corner, along_x, along_y))
    flap_mask = _polygon_mask(
        (roi_height, roi_width), (along_x, mirrored_corner, along_y)
    )

    values = window.astype(np.float32)
    values = _draw_flap(values, mirrored, flap_mask, paper, spec)
    values = _draw_gap(values, gap_mask, paper, spec)
    values = _draw_shadow(values, flap_mask, gap_mask, spec, margin)
    values = _draw_crease(values, along_x, along_y, paper, spec, active)

    window[:] = np.clip(values, 0, 255).astype(np.uint8)
    return result


def _bounding_roi(
    points: Sequence[Point], width: int, height: int, margin: int
) -> tuple[int, int, int, int] | None:
    """Return ``(left, top, width, height)`` covering ``points`` plus a margin.

    Clipped to the image. ``None`` when nothing of the fold is on the image at
    all, which keeps the caller from slicing an empty window.
    """
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    left = max(0, int(np.floor(min(xs))) - margin)
    top = max(0, int(np.floor(min(ys))) - margin)
    right = min(width, int(np.ceil(max(xs))) + margin + 1)
    bottom = min(height, int(np.ceil(max(ys))) + margin + 1)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _paper_level(window: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Estimate the paper's own tone in this part of the image, per channel."""
    if window.ndim == 3:
        return np.asarray(
            [
                float(np.percentile(window[..., index], _PAPER_PERCENTILE))
                for index in range(window.shape[2])
            ],
            dtype=np.float32,
        )
    return np.asarray(
        [float(np.percentile(window, _PAPER_PERCENTILE))], dtype=np.float32
    )


def _spread(
    value: NDArray[np.float32], like: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Shape a per-channel constant so it broadcasts over an image window."""
    if like.ndim == 3:
        return value.reshape(1, 1, -1)
    return value[:1]


def _supersample_factor(shape: tuple[int, int]) -> int:
    """How far to oversample a mask before reducing it.

    Anti-aliasing matters most where the shape is thinnest, which is exactly
    where the working area is smallest, so the factor falls away as the region
    grows. A severe fold on a 600 dpi scan does not need an eight-times canvas
    to get its edges right, and giving it one would cost tens of megabytes per
    sheet for no visible gain.
    """
    largest = max(shape)
    if largest <= 64:
        return 8
    if largest <= 256:
        return 4
    return 2


def _polygon_mask(shape: tuple[int, int], points: Sequence[Point]) -> NDArray[np.float32]:
    """Return a soft coverage mask in ``[0, 1]`` for one convex polygon.

    Anti-aliased, which is what lets a micro fold three pixels deep register as
    partial coverage rather than vanishing into rounding: ``fillConvexPoly``
    takes integer vertices, so a sub-pixel-thin triangle drawn directly would
    snap to nothing at all.
    """
    supersample = _supersample_factor(shape)
    canvas = np.zeros((shape[0] * supersample, shape[1] * supersample), dtype=np.uint8)
    polygon = np.array(
        [
            [round(point.x * supersample), round(point.y * supersample)]
            for point in points
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, polygon, 255, lineType=cv2.LINE_8)
    reduced = cv2.resize(canvas, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return np.asarray(reduced, dtype=np.float32) / 255.0


def _blend(
    values: NDArray[np.float32],
    replacement: NDArray[np.float32],
    mask: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Mix ``replacement`` into ``values`` where ``mask`` says so."""
    alpha = mask[..., np.newaxis] if values.ndim == 3 else mask
    return np.asarray(values * (1.0 - alpha) + replacement * alpha, dtype=np.float32)


def _draw_flap(
    values: NDArray[np.float32],
    mirrored: NDArray[np.uint8],
    flap_mask: NDArray[np.float32],
    paper: NDArray[np.float32],
    spec: FoldSpec,
) -> NDArray[np.float32]:
    """Lay the folded-back flap over the page.

    The flap is the reverse of the sheet: blank paper, slightly shaded because
    it is a second layer standing proud of the first, with the printing on its
    far side ghosting through as paper does.
    """
    blank = _spread(paper, values) * (1.0 - spec.flap_shade)
    ghost = mirrored.astype(np.float32)
    # Show-through darkens in proportion to how dark the hidden side is, so
    # blank areas of the flap stay blank and a marker behind it leaves a faint
    # smudge rather than a crisp copy.
    surface = blank - spec.show_through * np.clip(_spread(paper, values) - ghost, 0.0, None)
    return _blend(values, surface, flap_mask)


def _draw_gap(
    values: NDArray[np.float32],
    gap_mask: NDArray[np.float32],
    paper: NDArray[np.float32],
    spec: FoldSpec,
) -> NDArray[np.float32]:
    """Fill the triangle the corner vacated with what lies behind the sheet."""
    backing = _spread(paper, values) * (1.0 - spec.backing_shade)
    return _blend(values, backing, gap_mask)


def _draw_shadow(
    values: NDArray[np.float32],
    flap_mask: NDArray[np.float32],
    gap_mask: NDArray[np.float32],
    spec: FoldSpec,
    margin: int,
) -> NDArray[np.float32]:
    """Darken the page just outside the flap, where it stands off the paper.

    Built by blurring the flap's own mask and keeping only what spilled beyond
    it, which gives a soft rim that follows the flap's shape without any
    geometry of its own. Bounded by the blur kernel, so the fold stays local.
    """
    if spec.shadow <= 0.0:
        return values
    kernel = max(3, (int(margin) // 2) * 2 + 1)
    spread = cv2.GaussianBlur(flap_mask, (kernel, kernel), 0)
    rim = np.clip(spread - flap_mask - gap_mask, 0.0, 1.0) * spec.shadow
    alpha = rim[..., np.newaxis] if values.ndim == 3 else rim
    return np.asarray(values * (1.0 - alpha), dtype=np.float32)


def _draw_crease(
    values: NDArray[np.float32],
    first: Point,
    second: Point,
    paper: NDArray[np.float32],
    spec: FoldSpec,
    placement: PagePlacement,
) -> NDArray[np.float32]:
    """Draw the crease itself: the one hard line a fold leaves."""
    if spec.crease_darkness <= 0.0:
        return values
    thickness = max(
        1, round(0.0015 * min(placement.width, placement.height))
    )
    stroke = np.zeros(values.shape[:2], dtype=np.uint8)
    cv2.line(
        stroke,
        (round(first.x), round(first.y)),
        (round(second.x), round(second.y)),
        255,
        thickness,
        cv2.LINE_AA,
    )
    mask = np.asarray(stroke, dtype=np.float32) / 255.0 * spec.crease_darkness
    dark = _spread(paper, values) * 0.25
    return _blend(values, dark, mask)


def apply_corner_folds(
    image: NDArray[np.uint8],
    specs: Sequence[FoldSpec],
    placement: PagePlacement | None = None,
) -> NDArray[np.uint8]:
    """Apply several folds to one sheet, in corner order.

    Each is computed against the page, not against the result of the previous
    one, because two folded corners are two independent accidents - and because
    an order-dependent result would make a dataset's determinism depend on a
    detail nobody records. They stay apart by construction: a fold reaches at
    most :data:`MAX_DEPTH` of the page from its own corner, so two folds cannot
    meet in the middle.
    """
    if not specs:
        return image
    result = image
    for spec in sorted(specs, key=lambda item: CORNER_ORDER.index(item.corner)):
        result = apply_corner_fold(result, spec, placement)
    return result


__all__ = [
    "CORNER_ORDER",
    "MAX_DEPTH",
    "MIN_DEPTH",
    "SEVERITY_DEPTH_RANGES",
    "FoldCorner",
    "FoldSeverity",
    "FoldSpec",
    "PagePlacement",
    "apply_corner_fold",
    "apply_corner_folds",
    "crease",
    "fold_region",
    "fold_triangle",
    "overlap_fraction",
    "project_points",
    "reflection_matrix",
    "severity_for_depth",
]
