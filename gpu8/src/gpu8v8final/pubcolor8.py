"""Colour system of the publication figures, and a computable palette check.

The categorical order, the sequential blue ramp and the neutral midpoint are the
documented reference palette of the data-viz method (fixed order, never cycled;
colour follows the entity).  ``validate`` is a line-by-line Python port of the
method's ``validate_palette.js`` (OKLab Delta E x100, Machado-Oliveira-Fernandes
2009 CVD simulation at severity 1.0, WCAG contrast), so every palette actually
drawn is checked by computation rather than by eye.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- documented palette
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec", 350: "#5598e7",
        400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
        700: "#0d366b"}
NEUTRAL_MID = "#f0efec"          # diverging midpoint (light)
SURFACE = "#ffffff"              # print surface of the figures
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# entity -> categorical slot (fixed for every figure; colour follows the entity)
ENTITY = {
    # core trio + violet validate all-pairs (worst CVD dE 9.2, normal-vision 16.3): usable in any form
    "deployed": CAT[0],              # V8.1 deployed model
    "channel_lgb_v6": CAT[1],        # channel-only LightGBM on its own V6 500
    "v71_direct_strict": CAT[2],     # V7.1 direct fusion retrained on the strict 500
    "standalone_sF": CAT[6],         # teacher s_F alone (full background)
    # second-stage family (blue / magenta / yellow validate all-pairs); never drawn next to orange
    "B": CAT[3],                     # base ranker
    "corrector": CAT[4],             # gated corrector
    # context series: neutral ink (identity by label / marker, not hue)
    "v71_direct_v6": "#6f6e69",
    "F7": "#a8a7a1",
    "kdgene": "#6f6e69",
    "sign": "#a8a7a1",
}

# ---------------------------------------------------------------- validator (port)
BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
ORDINAL_MIN_DL = 0.06
ORDINAL_LIGHT_FLOOR = 2.0
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h: str) -> list[float]:
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin2s(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def lin(h: str) -> list[float]:
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h: str) -> float:
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb) -> list[float]:
    r, g, b = rgb
    l_ = math.copysign(abs(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
                       0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = math.copysign(abs(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
                       0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = math.copysign(abs(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3),
                       0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return [0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_]


def oklab(h: str) -> list[float]:
    return oklab_from_lin(lin(h))


def oklch(h: str) -> tuple[float, float]:
    L, a, b = oklab(h)
    return L, math.hypot(a, b)


def okhue(h: str) -> float:
    _, a, b = oklab(h)
    return (math.degrees(math.atan2(b, a)) % 360 + 360) % 360


def simulate(h: str, kind: str) -> list[float]:
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [max(0.0, min(1.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def validate(palette: list[str], mode: str = "light", surface: str = SURFACE, pairs: str = "adjacent") -> dict:
    lo, hi = BAND[mode]
    rep, ok = [], True
    off = [(c, round(oklch(c)[0], 3)) for c in palette if not lo <= oklch(c)[0] <= hi]
    ok &= not off
    rep.append(("Lightness band", "PASS" if not off else "FAIL", off or f"all {len(palette)} inside L {lo}-{hi}"))
    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    rep.append(("Chroma floor", "PASS" if not lowc else "FAIL", lowc or f"all >= {CHROMA_FLOOR}"))
    n = len(palette)
    pl = ([(i, j) for i in range(n) for j in range(i + 1, n)] if pairs == "all"
          else [(i, i + 1) for i in range(n - 1)])
    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pl:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tri = min((delta_e(palette[i], palette[j], "tritan") for i, j in pl), default=99)
    wd = worst[0] if worst else 99
    state = "PASS" if wd >= CVD_TARGET else ("WARN" if wd >= CVD_FLOOR else "FAIL")
    ok &= state != "FAIL"
    rep.append(("CVD separation", state, f"worst {pairs} {worst[2]}<->{worst[3]} dE {wd:.1f} ({worst[1]}); "
                                         f"tritan {tri:.1f}" if worst else "n/a"))
    nw = None
    for i, j in pl:
        d = delta_e(palette[i], palette[j])
        if nw is None or d < nw[0]:
            nw = (d, palette[i], palette[j])
    nd = nw[0] if nw else 99
    ok &= nd >= NORMAL_FLOOR
    rep.append(("Normal-vision floor", "PASS" if nd >= NORMAL_FLOOR else "FAIL",
                f"worst {pairs} {nw[1]}<->{nw[2]} dE {nd:.1f}" if nw else "n/a"))
    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    rep.append(("Contrast vs surface", "WARN" if low else "PASS",
                f"relief required (visible labels / source-data table): {low}" if low else f"all >= {CONTRAST_MIN}:1"))
    return {"palette": palette, "mode": mode, "surface": surface, "pairs": pairs, "ok": bool(ok),
            "report": [list(r) for r in rep]}


def validate_ordinal(palette: list[str], mode: str = "light", surface: str = SURFACE) -> dict:
    rep, ok = [], True
    Ls = [oklch(c)[0] for c in palette]
    order = sorted(range(len(Ls)), key=lambda i: Ls[i])
    mono = order == list(range(len(Ls))) or order == list(range(len(Ls)))[::-1]
    ok &= mono
    rep.append(("Lightness monotone", "PASS" if mono else "FAIL", [round(x, 3) for x in Ls]))
    gaps = [abs(Ls[i + 1] - Ls[i]) for i in range(len(Ls) - 1)]
    thin = [(palette[i], palette[i + 1], round(g, 3)) for i, g in enumerate(gaps) if g < ORDINAL_MIN_DL]
    ok &= not thin
    rep.append(("Adjacent dL", "PASS" if not thin else "FAIL", thin or f"all gaps >= {ORDINAL_MIN_DL}"))
    byL = sorted(palette, key=lambda c: oklch(c)[0])
    lightest = byL[-1] if mode == "light" else byL[0]
    cr = contrast(lightest, surface)
    ok &= cr >= ORDINAL_LIGHT_FLOOR
    rep.append(("Light-end contrast", "PASS" if cr >= ORDINAL_LIGHT_FLOOR else "FAIL", f"{lightest} {cr:.2f}:1"))
    hues = [okhue(c) for c in palette]
    spread = max(hues) - min(hues) if hues else 0
    spread = 360 - spread if spread > 180 else spread
    ok &= spread <= 40
    rep.append(("Single hue", "PASS" if spread <= 40 else "FAIL", f"hue spread {spread:.0f} deg"))
    return {"palette": palette, "mode": mode, "surface": surface, "ordinal": True, "ok": bool(ok),
            "report": [list(r) for r in rep]}


# ---------------------------------------------------------------- derived scales
def _mix_oklab(h1: str, h2: str, t: float) -> str:
    """Interpolate two documented anchors in OKLab (used only to build continuous colormaps)."""
    a, b = np.array(oklab(h1)), np.array(oklab(h2))
    L, A, B = a + (b - a) * t
    l_ = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3
    m_ = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3
    s_ = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bb = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    return "#" + "".join(f"{round(lin2s(c) * 255):02x}" for c in (r, g, bb))


def seq_cmap(name: str = "pub_blue"):
    from matplotlib.colors import LinearSegmentedColormap
    steps = [BLUE[k] for k in sorted(BLUE)]
    return LinearSegmentedColormap.from_list(name, ["#f4f8fd"] + steps, N=256)


def div_cmap(name: str = "pub_div"):
    """Blue <-> red with a neutral grey midpoint (equal arms, OKLab interpolation of documented anchors)."""
    from matplotlib.colors import LinearSegmentedColormap
    neg = [_mix_oklab(BLUE[650], NEUTRAL_MID, t) for t in np.linspace(0, 1, 6)]
    pos = [_mix_oklab(NEUTRAL_MID, CAT[7], t) for t in np.linspace(0, 1, 6)[1:]]
    return LinearSegmentedColormap.from_list(name, neg + pos, N=256)


def div_cmap_rev(name: str = "pub_div_r"):
    """Red (negative) <-> blue (positive): positive deltas read as gains in blue."""
    from matplotlib.colors import LinearSegmentedColormap
    neg = [_mix_oklab(CAT[7], NEUTRAL_MID, t) for t in np.linspace(0, 1, 6)]
    pos = [_mix_oklab(NEUTRAL_MID, BLUE[650], t) for t in np.linspace(0, 1, 6)[1:]]
    return LinearSegmentedColormap.from_list(name, neg + pos, N=256)


def ink_on(fill: str) -> str:
    """Ink or white for a label set inside a fill, whichever clears contrast."""
    return "#ffffff" if contrast("#ffffff", fill) >= contrast(INK, fill) else INK


def run_checks() -> dict:
    """Validate every categorical combination the figures actually draw (written to the figure manifest).
    Neutral greys carry context series and are identified by labels/markers, so they are not part of these sets."""
    E = ENTITY
    out = {
        "documented_8_adjacent": validate(CAT, pairs="adjacent"),
        # Fig 3a-d, 5a-b, 4a (with violet for the semantic order), S7a: the four method entities, any two may touch
        "methods_4_all_pairs": validate([E["deployed"], E["channel_lgb_v6"], E["v71_direct_strict"], E["standalone_sF"]],
                                        pairs="all"),
        # Fig 6, S2c: deployed/A, corrector, base ranker
        "second_stage_all_pairs": validate([E["deployed"], E["corrector"], E["B"]], pairs="all"),
        # Fig 9a: comparator blocks in drawn order (orange, aqua, yellow)
        "external_blocks_adjacent": validate([E["channel_lgb_v6"], E["v71_direct_strict"], E["B"]], pairs="adjacent"),
        # Fig 9c: bar order blue, orange, violet, yellow
        "historical_bars_adjacent": validate([E["deployed"], E["channel_lgb_v6"], E["standalone_sF"], E["B"]],
                                             pairs="adjacent"),
        # Fig 3f, 4d, S7a: gain (blue) vs loss (orange)
        "gain_loss_pair": validate([E["deployed"], E["channel_lgb_v6"]], pairs="all"),
        # ordinal blue ramp used for Fig 1e and the alluvial nodes
        "blue_ordinal_ramp": validate_ordinal([BLUE[k] for k in (250, 450, 650)]),
    }
    return out
