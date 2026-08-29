import math, random
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation

SIZE = 1024
CENTER = np.array([SIZE / 2, SIZE / 2])
CORNER_DIST = math.hypot(SIZE / 2, SIZE / 2) + 20
GOLDEN_ANGLE = 137.508
C = 12.0
N_POINTS = 4200
BG_COLOR = (28, 3, 34)

random.seed(7)
np.random.seed(7)

R_CORE, R_YELLOW1, R_ORANGE1, R_YELLOW2, R_ORANGE2, R_WHITE, R_YELLOW3, R_ORANGE3 = \
    22, 60, 120, 190, 270, 330, 410, 480

ZONE_COLORS = {
    "core":    [(255, 250, 235)],
    "yellow1": [(255, 205, 60)],
    "orange1": [(255, 82, 0)],
    "yellow2": [(255, 188, 0)],
    "orange2": [(214, 60, 5)],
    "white":   [(255, 253, 245)],
    "yellow3": [(255, 197, 10)],
    "orange3": [(185, 50, 5)],
    "purple":  [(92, 12, 128)],
}

# unified warm-gold gradient for Maveli's whole body, instead of coloring
# each pixel by whichever concentric ring happens to sit behind it (that
# ring-banding is what was reading as "messy" — the torso looked sliced by
# the rings passing through it)
FIGURE_TOP_COLOR = (255, 230, 150)     # bright warm gold near the collar/chest
FIGURE_BOTTOM_COLOR = (140, 40, 15)    # deep grounded maroon near the feet


def figure_fill_color(y):
    t = np.clip((y - _fig_top_y) / max(1, (_fig_bot_y - _fig_top_y)), 0, 1)
    return tuple(int(FIGURE_TOP_COLOR[c] + (FIGURE_BOTTOM_COLOR[c] - FIGURE_TOP_COLOR[c]) * t)
                 for c in range(3))
ZONE_SIZE = {"core": 16, "yellow1": 20, "orange1": 22, "yellow2": 22,
             "orange2": 22, "white": 22, "yellow3": 22, "orange3": 22, "purple": 24}
ZONE_ORDER = [
    (R_CORE, "core"), (R_YELLOW1, "yellow1"), (R_ORANGE1, "orange1"),
    (R_YELLOW2, "yellow2"), (R_ORANGE2, "orange2"), (R_WHITE, "white"),
    (R_YELLOW3, "yellow3"), (R_ORANGE3, "orange3"), (float("inf"), "purple"),
]

# ---- Traced Maveli + umbrella silhouette (flattened SVG polygon) ----
MAVELI_POLY = np.load("maveli_poly.npy")
LOCAL_W = int(MAVELI_POLY[:, 0].max()) + 2
LOCAL_H = int(MAVELI_POLY[:, 1].max()) + 2

SCALE = 0.78

_yy0, _xx0 = np.mgrid[0:LOCAL_H, 0:LOCAL_W]
_mpl_path0 = MplPath(MAVELI_POLY)
_flat_pts0 = np.column_stack([_xx0.ravel(), _yy0.ravel()])
_inside0 = _mpl_path0.contains_points(_flat_pts0).reshape(LOCAL_H, LOCAL_W)
_umbrella_local0 = _inside0 & (_xx0 > 140) & (_yy0 < 225)

# the collar/neck notch shares this same coordinate range as the umbrella
# pole in the traced path — carve it out explicitly using its real vertices
# (from the original SVG path) so purple stops bleeding into the head/collar
_collar_pts = np.array([
    (136, 26.5), (155.5, 95), (196.5, 132), (219.5, 151.5), (164, 243),
    (155.5, 230.5), (158.5, 204), (172, 165), (150.5, 132), (111.5, 120),
    (90, 140.5), (136, 26.5),
])
_collar_path = MplPath(_collar_pts)
_collar_mask0 = _collar_path.contains_points(_flat_pts0).reshape(LOCAL_H, LOCAL_W)
_umbrella_local0 = _umbrella_local0 & ~_collar_mask0

_figure_local0 = _inside0 & ~_umbrella_local0

# center on the BODY specifically (what the eye reads as "Maveli"), not the
# combined bbox — the umbrella's rightward mass was pulling bbox-center
# calculations left of where the body actually needs to sit
_local_cx = _xx0[_figure_local0].mean()
_local_cy = _yy0[_figure_local0].mean()
OFFSET_X = int(SIZE / 2 - _local_cx * SCALE)
OFFSET_Y = int(SIZE / 2 - _local_cy * SCALE)

_yy, _xx = _yy0, _xx0
_inside = _inside0
_umbrella_local = _umbrella_local0
_figure_local = _figure_local0

_region = np.zeros((LOCAL_H, LOCAL_W), dtype=np.uint8)
_region[_figure_local] = 2
_region[_umbrella_local] = 1

_scaled_w = int(LOCAL_W * SCALE)
_scaled_h = int(LOCAL_H * SCALE)
_region_img = Image.fromarray(_region).resize((_scaled_w, _scaled_h), Image.NEAREST)
_region_scaled = np.array(_region_img)

_canvas_region = np.zeros((SIZE, SIZE), dtype=np.uint8)
_px0, _py0 = OFFSET_X, OFFSET_Y
_px1 = min(SIZE, _px0 + _scaled_w)
_py1 = min(SIZE, _py0 + _scaled_h)
_sx0, _sy0 = max(0, -_px0), max(0, -_py0)
_canvas_region[max(0, _py0):_py1, max(0, _px0):_px1] = \
    _region_scaled[_sy0:_sy0 + (_py1 - max(0, _py0)), _sx0:_sx0 + (_px1 - max(0, _px0))]

UMBRELLA_MASK = _canvas_region == 1
FIGURE_MASK = _canvas_region == 2

# fix: the traced silhouette's umbrella/figure split (x > 140) occasionally
# cuts off a small disconnected sliver right next to the head. Keep only
# the umbrella's real connected shape as purple; fold any stray fragment
# into the figure instead, since visually it sits against his head/collar,
# not the parasol. Remember its centroid — that's the natural collar point
# where the handle should actually land.
from scipy.ndimage import label as _cc_label
_lab, _n_cc = _cc_label(UMBRELLA_MASK)
_collar_anchor = None
if _n_cc > 1:
    _sizes = [(_lab == i).sum() for i in range(1, _n_cc + 1)]
    _main_id = int(np.argmax(_sizes)) + 1
    _stray = UMBRELLA_MASK & (_lab != _main_id)
    _sys_, _sxs_ = np.where(_stray)
    if len(_sys_):
        _collar_anchor = (int(_sxs_.mean()), int(_sys_.mean()))
    UMBRELLA_MASK = UMBRELLA_MASK & (_lab == _main_id)
    FIGURE_MASK = FIGURE_MASK | _stray

# fix: the traced silhouette also includes a thin, jagged natural "pole"
# connecting the dome down to the collar. It's real geometry (not noise),
# but at flower-sprite scale a thin wiggly line reads as scattered stray
# purple stipples rather than a clean stick. Strip thin filaments with a
# morphological opening (keeps the solid dome cap, removes anything
# narrower than the opening radius), then draw one clean straight handle
# explicitly instead of relying on the jagged trace.
from scipy.ndimage import binary_opening as _binary_opening
_opened = _binary_opening(UMBRELLA_MASK, structure=np.ones((13, 13)))
# reclaim whatever the opening stripped away (the thin pole) as normal
# background so it doesn't render as leftover purple specks
UMBRELLA_MASK = _opened

_fig_ys_c, _fig_xs_c = np.where(FIGURE_MASK)
_fig_top_y, _fig_bot_y = (int(_fig_ys_c.min()), int(_fig_ys_c.max())) if len(_fig_ys_c) else (0, SIZE)

# fix: the traced silhouette's feet are asymmetric — the left foot has a
# natural rounded heel/toe bump, the right is basically a flat vertical
# cut. Mirror the bottom ~22% of the figure (the foot region) across its
# own horizontal centerline and OR it with the original, so whichever
# side is fuller "wins" on both sides — in practice this gives the right
# foot the left foot's rounded shape instead of looking cut off.
if len(_fig_xs_c):
    _center_x = int(round((_fig_xs_c.min() + _fig_xs_c.max()) / 2))
    _foot_y0 = int(_fig_bot_y - 0.22 * (_fig_bot_y - _fig_top_y))
    _xs_row = np.arange(SIZE)
    _mirror_xs = np.clip(2 * _center_x - _xs_row, 0, SIZE - 1)
    for _yv in range(max(0, _foot_y0), min(SIZE, _fig_bot_y + 1)):
        _row = FIGURE_MASK[_yv]
        _mirrored = _row[_mirror_xs]
        FIGURE_MASK[_yv] = _row | _mirrored


def segment_mask(xx, yy, x0, y0, x1, y1, width):
    ex, ey = x1 - x0, y1 - y0
    seg_len2 = ex * ex + ey * ey
    px, py = xx - x0, yy - y0
    t = np.clip((px * ex + py * ey) / seg_len2, 0, 1)
    projx, projy = x0 + t * ex, y0 + t * ey
    dist = np.sqrt((xx - projx) ** 2 + (yy - projy) ** 2)
    return dist <= width


# draw one clean, straight handle: from the umbrella's actual tip/edge
# nearest the head, to a point specifically on the RIGHT side of Maveli's
# neck — not "wherever the dome's lowest pixel happens to be" (which could
# land mid-canopy on a tilted ellipse) and not "wherever the nearest head
# pixel happens to be" (which could land on top of the skull instead of
# the neck).
_dome_ys, _dome_xs = np.where(UMBRELLA_MASK)
HANDLE_MASK = np.zeros((SIZE, SIZE), dtype=bool)
if len(_dome_ys) and len(_fig_ys_c):
    # 1. Find the neck attachment point first: within the head+neck band
    #    (top ~28% of the figure), take the row about halfway down that
    #    band (skull top and shoulder both excluded) and use its
    #    RIGHT-most silhouette pixel — i.e. the actual right edge of the
    #    neck, not an interior point.
    _neck_y_hi = _fig_top_y + 0.42 * (_fig_bot_y - _fig_top_y)
    _neck_y_target = int(_fig_top_y + 0.32 * (_fig_bot_y - _fig_top_y))
    _dx1 = _dy1 = None
    for _yv in range(_neck_y_target, int(_neck_y_hi) + 1):
        _row_xs = np.where(FIGURE_MASK[_yv])[0]
        if len(_row_xs):
            _dx1, _dy1 = int(_row_xs.max()), _yv
            break
    if _dx1 is None:
        # fallback: nearest figure pixel to the umbrella overall
        _nys, _nxs = np.where(FIGURE_MASK)
        _dx1, _dy1 = int(_nxs[0]), int(_nys[0])

    # 2. Now find the umbrella's own boundary point CLOSEST to that neck
    #    point — this is guaranteed to land on the shape's actual edge
    #    (its "shell tip") rather than an arbitrary interior pixel,
    #    because it's the point of nearest approach between two shapes.
    _umb_eroded = binary_erosion(UMBRELLA_MASK, iterations=1)
    _umb_boundary = UMBRELLA_MASK & ~_umb_eroded
    _bys, _bxs = np.where(_umb_boundary)
    _d2 = (_bxs - _dx1) ** 2 + (_bys - _dy1) ** 2
    _nearest = np.argmin(_d2)
    _dx0, _dy0 = int(_bxs[_nearest]), int(_bys[_nearest])

    _cyy, _cxx = np.mgrid[0:SIZE, 0:SIZE]
    HANDLE_MASK = segment_mask(_cxx, _cyy, _dx0, _dy0, _dx1, _dy1, 1.8)
    # allow the handle to overlap slightly into the figure's edge
    # (erode the figure mask before excluding) so the stick visually
    # reaches all the way to the hand instead of stopping short right
    # at the boundary and leaving a gap that other passes then fill
    # with the wrong color
    _figure_eroded_for_handle = binary_erosion(FIGURE_MASK, iterations=3)
    HANDLE_MASK = HANDLE_MASK & ~_figure_eroded_for_handle
    UMBRELLA_MASK = UMBRELLA_MASK | HANDLE_MASK

_SILHOUETTE_MASK = UMBRELLA_MASK | FIGURE_MASK
_EDGE_DIST = distance_transform_edt(_SILHOUETTE_MASK)

# tighter halo (was 8 px, read as a blurry messy aura) -> crisp 3px band
_dilated = binary_dilation(_SILHOUETTE_MASK, iterations=3)
_BORDER_MASK = _dilated & ~_SILHOUETTE_MASK

# the stick gets its OWN thin silver outline (dedicated, proportionate to
# its width) instead of being fully excluded from silver — a bare stick
# with no trim reads as "just floating there", not as a deliberately
# designed part of the piece
_HANDLE_BORDER = binary_dilation(HANDLE_MASK, iterations=6) & ~HANDLE_MASK
_HANDLE_BORDER = _HANDLE_BORDER & ~FIGURE_MASK
# keep the silver halo off the thin stick entirely — on a line this narrow
# the border was as wide as the stick itself, visually swallowing the
# purple. Erase any halo pixels close to the handle so it stays solid.
_handle_halo_zone = binary_dilation(HANDLE_MASK, iterations=4)
_BORDER_MASK = _BORDER_MASK & ~_handle_halo_zone


def crescent_open(x, y):
    ang = math.degrees(math.atan2(-(y - CENTER[1]), x - CENTER[0])) % 360
    return 25 <= ang <= 70


def pick_zone(x, y, r):
    ix, iy = int(x), int(y)
    in_umbrella = in_figure = False
    if 0 <= ix < SIZE and 0 <= iy < SIZE:
        in_umbrella = bool(UMBRELLA_MASK[iy, ix])
        in_figure = bool(FIGURE_MASK[iy, ix])

    if in_umbrella:
        return "purple", False, False

    zone = "purple"
    for boundary, z in ZONE_ORDER:
        if r < boundary:
            zone = z
            break
    if zone == "white" and crescent_open(x, y):
        zone = "orange2"

    near_edge = in_figure and 0 <= ix < SIZE and 0 <= iy < SIZE and _EDGE_DIST[iy, ix] < 6
    return zone, in_figure, near_edge


def make_flower_sprite(radius, base_rgb, seed, is_center_dot=False, accent_center=False):
    rng = random.Random(seed)
    R = radius * rng.uniform(0.95, 1.05)
    dim = int(R * 2.6) + 4
    cx = cy = dim / 2

    img = Image.new("RGBA", (dim, dim), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    n_petals = rng.randint(6, 8)
    phase = rng.uniform(0, 2 * math.pi)
    petal_len = R * 1.05
    petal_w = R * 0.50

    darker = tuple(max(0, int(c * 0.72)) for c in base_rgb)

    for i in range(n_petals):
        ang = phase + i * (2 * math.pi / n_petals)
        perp = ang + math.pi / 2
        base_off = 0.10 * petal_len
        bx, by = cx + base_off * math.cos(ang), cy + base_off * math.sin(ang)
        b1 = (bx + petal_w * 0.5 * math.cos(perp), by + petal_w * 0.5 * math.sin(perp))
        b2 = (bx - petal_w * 0.5 * math.cos(perp), by - petal_w * 0.5 * math.sin(perp))
        mid = 0.6
        m1 = (cx + petal_len * mid * math.cos(ang) + petal_w * 0.42 * math.cos(perp),
              cy + petal_len * mid * math.sin(ang) + petal_w * 0.42 * math.sin(perp))
        m2 = (cx + petal_len * mid * math.cos(ang) - petal_w * 0.42 * math.cos(perp),
              cy + petal_len * mid * math.sin(ang) - petal_w * 0.42 * math.sin(perp))
        tip = (cx + petal_len * math.cos(ang), cy + petal_len * math.sin(ang))
        pts = [b1, m1, tip, m2, b2]
        fill = base_rgb if i % 2 == 0 else darker
        outline_tone = tuple(max(0, int(c * 0.45)) for c in fill)
        draw.polygon(pts, fill=(*fill, 255), outline=(*outline_tone, 150))

    center_r = R * 0.26
    if accent_center:
        center_color = (232, 232, 238, 255)   # silver, not gold
    elif is_center_dot:
        center_color = (200, 20, 20, 255)
    else:
        center_color = (*tuple(max(0, c - 75) for c in base_rgb), 255)
    center_outline = tuple(max(0, int(c * 0.5)) for c in center_color[:3])
    draw.ellipse([cx - center_r, cy - center_r, cx + center_r, cy + center_r],
                 fill=center_color, outline=(*center_outline, 130))

    return img, dim


def make_pearl_sprite(radius, color, seed):
    """Clean, round, soft-gradient bead — used for the silhouette piping and
    ring borders instead of the jagged star-petal sprite, so the trim reads
    as a deliberate strung-pearl edge instead of scattered messy dots."""
    rng = random.Random(seed)
    R = radius * rng.uniform(0.96, 1.04)
    dim = int(R * 2.4) + 4
    cx = cy = dim / 2
    ys, xs = np.mgrid[0:dim, 0:dim]
    rad = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    alpha = np.clip((R - rad) / R, 0, 1) ** 0.55
    shade = 0.72 + 0.34 * alpha
    color_arr = np.empty((dim, dim, 3), dtype=np.float32)
    for c in range(3):
        color_arr[:, :, c] = np.clip(color[c] * shade, 0, 255)

    highlight_r = R * 0.32
    hx, hy = cx - R * 0.28, cy - R * 0.28
    hl_d = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2)
    hl = np.clip((highlight_r - hl_d) / highlight_r, 0, 1) * 0.5
    for c in range(3):
        color_arr[:, :, c] = np.clip(color_arr[:, :, c] + 255 * hl, 0, 255)

    color_arr = color_arr.astype(np.uint8)
    alpha_arr = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([color_arr, alpha_arr]), "RGBA"), dim


def generate():
    canvas = Image.new("RGBA", (SIZE, SIZE), (*BG_COLOR, 255))
    points, n = [], 0
    while True:
        theta = math.radians(n * GOLDEN_ANGLE)
        r = C * math.sqrt(n)
        if r > CORNER_DIST or n > N_POINTS:
            break
        x = CENTER[0] + r * math.cos(theta)
        y = CENTER[1] + r * math.sin(theta)
        points.append((n, x, y, r))
        n += 1
    points.sort(key=lambda p: -p[3])

    for n, x, y, r in points:
        if x < -30 or x > SIZE + 30 or y < -30 or y > SIZE + 30:
            continue
        zone, in_figure, near_edge = pick_zone(x, y, r)
        if in_figure:
            base_rgb = figure_fill_color(y)
        else:
            base_rgb = ZONE_COLORS[zone][0]
        radius = ZONE_SIZE[zone]
        sprite, dim = make_flower_sprite(radius, base_rgb, seed=n,
                                          is_center_dot=(n == 0), accent_center=near_edge)
        canvas.alpha_composite(sprite, (int(x - dim / 2), int(y - dim / 2)))

    # dedicated fill pass so the umbrella (a small, thin shape) is fully covered
    step = 7
    y_lo, y_hi = OFFSET_Y - 20, OFFSET_Y + int(LOCAL_H * SCALE) + 20
    x_lo, x_hi = OFFSET_X - 20, OFFSET_X + int(LOCAL_W * SCALE) + 20
    for yv in range(max(0, y_lo), min(SIZE, y_hi), step):
        for xv in range(max(0, x_lo), min(SIZE, x_hi), step):
            if UMBRELLA_MASK[yv, xv] and not HANDLE_MASK[yv, xv]:
                sprite, dim = make_flower_sprite(20, ZONE_COLORS["purple"][0], seed=yv * 4096 + xv)
                canvas.alpha_composite(sprite, (int(xv - dim / 2), int(yv - dim / 2)))
            elif FIGURE_MASK[yv, xv]:
                # fill any gaps left by the sparse main spiral with the
                # SAME unified gold gradient as the main pass, so the body
                # reads as one coherent figure, not ring-sliced patchwork
                base_rgb = figure_fill_color(yv)
                near_edge = _EDGE_DIST[yv, xv] < 6
                sprite, dim = make_flower_sprite(19, base_rgb, seed=yv * 4096 + xv + 1, accent_center=near_edge)
                canvas.alpha_composite(sprite, (int(xv - dim / 2), int(yv - dim / 2)))

    # extra fine-grained pass just for the thin handle (a 7px grid can miss
    # a ~3px-wide line entirely, and the dome's radius-20 sprites made it
    # look chunky) — redraw it on top with thinner, tightly packed sprites
    # so it reads as a slim rod instead of matching the dome's bulk
    _hys, _hxs = np.where(HANDLE_MASK)
    if len(_hys):
        for _yv in range(int(_hys.min()), int(_hys.max()) + 1, 2):
            for _xv in range(int(_hxs.min()), int(_hxs.max()) + 1, 2):
                if HANDLE_MASK[_yv, _xv]:
                    sprite, dim = make_flower_sprite(4, ZONE_COLORS["purple"][0], seed=_yv * 4096 + _xv + 7)
                    canvas.alpha_composite(sprite, (int(_xv - dim / 2), int(_yv - dim / 2)))

    # silver "piping" halo around Maveli's silhouette — dense, overlapping
    # pearls so the trim reads as one solid continuous silver line rather
    # than separated dots with visible gaps between them.
    PIPING = (230, 230, 238)

    # dedicated thin silver outline for the stick itself — small sprites,
    # tight spacing, proportionate to the stick's own width (unlike the
    # thick general silhouette halo, which was excluded from this area)
    _hbys, _hbxs = np.where(_HANDLE_BORDER)
    for i in range(0, len(_hbys), 1):
        yv, xv = int(_hbys[i]), int(_hbxs[i])
        sprite, dim = make_flower_sprite(7, PIPING, seed=150000 + i)
        canvas.alpha_composite(sprite, (int(xv - dim / 2), int(yv - dim / 2)))

    _byy, _bxx = np.where(_BORDER_MASK)
    if len(_byy):
        _bmin_y, _bmax_y = _byy.min(), _byy.max()
        _bmin_x, _bmax_x = _bxx.min(), _bxx.max()
        border_spacing = 5   # was 8: tighter packing closes the gaps
        for yv in range(_bmin_y, _bmax_y + 1, border_spacing):
            row_offset = 2 if (yv // border_spacing) % 2 else 0
            for xv in range(_bmin_x + row_offset, _bmax_x + 1, border_spacing):
                if _BORDER_MASK[yv, xv]:
                    sprite, dim = make_flower_sprite(11, PIPING, seed=yv * 4096 + xv + 2)
                    canvas.alpha_composite(sprite, (int(xv - dim / 2), int(yv - dim / 2)))

    # thin silver piping rings between each concentric color zone boundary,
    # packed densely enough along the circumference to form solid rings
    ring_boundaries = [R_CORE, R_YELLOW1, R_ORANGE1, R_YELLOW2, R_ORANGE2,
                       R_WHITE, R_YELLOW3, R_ORANGE3]
    for rb in ring_boundaries:
        circumference = 2 * math.pi * rb
        n_dots = max(24, int(circumference / 5.5))
        # zigzag amplitude/frequency: bigger rings get a slightly wider
        # zigzag so the scallop stays proportionate, small rings stay subtle
        zig_amp = max(4, min(11, rb * 0.03))
        zig_freq = max(10, int(circumference / 34))  # petal-like tooth count
        for i in range(n_dots):
            ang = 2 * math.pi * i / n_dots
            # triangle wave, not sine, so it reads as an angular zigzag
            # (petal teeth) rather than a soft wobble
            phase = (zig_freq * ang / (2 * math.pi)) % 1.0
            tri = 4 * abs(phase - 0.5) - 1  # -1..1 triangle wave
            r_here = rb + zig_amp * tri
            xv = CENTER[0] + r_here * math.cos(ang)
            yv = CENTER[1] + r_here * math.sin(ang)
            ix, iy = int(xv), int(yv)
            if not (0 <= ix < SIZE and 0 <= iy < SIZE):
                continue
            if _SILHOUETTE_MASK[iy, ix] or UMBRELLA_MASK[iy, ix]:
                continue
            sprite, dim = make_flower_sprite(9, PIPING, seed=int(rb) * 999 + i)
            canvas.alpha_composite(sprite, (int(xv - dim / 2), int(yv - dim / 2)))

    # a crisp thin silver contour hugging the exact silhouette edge (1px
    # erosion-diff), drawn on top, so the outline itself is razor sharp
    # rather than relying only on the halo band
    _sil_eroded = binary_erosion(_SILHOUETTE_MASK, iterations=2)
    _sil_contour = _SILHOUETTE_MASK & ~_sil_eroded
    _sil_contour = _sil_contour & ~_handle_halo_zone  # keep silver off the thin stick here too
    _cyy, _cxx = np.where(_sil_contour)
    for i in range(0, len(_cyy), 3):
        yv, xv = int(_cyy[i]), int(_cxx[i])
        # tiny alternating in/out jitter along the contour itself, so the
        # silhouette edge reads as a scalloped trim, not a wire outline
        jig = 3 if (i // 3) % 2 == 0 else -3
        nx = xv + jig * math.cos(i * 0.7)
        ny = yv + jig * math.sin(i * 0.7)
        sprite, dim = make_flower_sprite(6, PIPING, seed=90000 + i)
        canvas.alpha_composite(sprite, (int(nx - dim / 2), int(ny - dim / 2)))

    # thin outer gold trim ring — a deliberate "medal edge" finishing touch
    GOLD = (255, 205, 90)
    trim_r = R_ORANGE3 + 34
    trim_circ = 2 * math.pi * trim_r
    n_trim = int(trim_circ / 6)
    for i in range(n_trim):
        ang = 2 * math.pi * i / n_trim
        wobble = 5 * math.sin(ang * 40)
        xv = CENTER[0] + (trim_r + wobble) * math.cos(ang)
        yv = CENTER[1] + (trim_r + wobble) * math.sin(ang)
        sprite, dim = make_flower_sprite(8, GOLD, seed=77000 + i)
        canvas.alpha_composite(sprite, (int(xv - dim / 2), int(yv - dim / 2)))

    # final guaranteed redraw of the handle, on top of every other pass
    # (silver piping/contour), so the stick is always visibly solid purple
    _hys2, _hxs2 = np.where(HANDLE_MASK)
    if len(_hys2):
        for _yv in range(int(_hys2.min()), int(_hys2.max()) + 1, 2):
            for _xv in range(int(_hxs2.min()), int(_hxs2.max()) + 1, 2):
                if HANDLE_MASK[_yv, _xv]:
                    sprite, dim = make_flower_sprite(4, ZONE_COLORS["purple"][0], seed=_yv * 4096 + _xv + 71)
                    canvas.alpha_composite(sprite, (int(_xv - dim / 2), int(_yv - dim / 2)))

    canvas.convert("RGB").save("pookalam_1024.png")


if __name__ == "__main__":
    generate()
