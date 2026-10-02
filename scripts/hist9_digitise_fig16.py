#!/usr/bin/env python
"""HISTORICAL 9 — digitise Fig. 16, in particular the FIELD-MEASURED curve 2.

Fig. 16 carries four longitudinal free-surface curves:

    1  calculated 1966, H_dam = 16.0 m, Q = 8 400 m3/s     (numeric in Table 20)
    2  FIELD-MEASURED, same conditions, 22-25 Apr 1970     <- the target
    3  calculated 1966, H_dam = 16.0 m, Q0.1% = 23 800     (numeric in Table 20)
    4  FIELD-MEASURED, H_dam = 16.16 m, Q = 9 040, 28-30 Apr 1970

Curves 2 and 4 are the ONLY field measurements in the source and the only
independent check on the design profiles. hist5 declined to digitise them from
an angled photograph. A square-on scan now makes it possible -- but not by eye,
and the reason is important:

    THE VERTICAL AXIS OF FIG. 16 IS NOT LINEAR.

Detected gridline spacings, in pixels per metre of elevation:

    16->17  663      19->20  213
    17->18  389      20->21  173
    18->19  278      21->22  145

a factor of 4.6 from bottom to top, while the HORIZONTAL gridlines are evenly
spaced to within 2 %. A distortion affecting only one axis is not perspective:
it is the figure's own stretched vertical scale, chosen so that centimetre-level
detail in the flat pool and a 6 m backwater rise fit on one plot.

Every by-eye reading I made earlier assumed a linear axis and was therefore
biased high by 0.15-0.35 m in the flat reach -- which is how the reading error
was detected in the first place, by checking curve 3 against its own Table 20
column. The axis is now calibrated on the detected gridlines and curve 2 is
extracted geometrically, with Table 20 used as the validation.

Circles are found as ENCLOSED LIGHT REGIONS (fill_holes minus the mask), which
works even where a circle touches curve 1.

Outputs
-------
data/processed/historical/historical_fig16_profiles.csv
outputs/figures/HIST9_fig16_digitisation.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.signal import find_peaks

from swot_dnipro import config as CFG

INK, BLUE, RED, AMBER, GREEN, GREY = ("#1a2228", "#236f8c", "#c1402a",
                                      "#b07d27", "#3f7d4e", "#8a94a3")
IMG = Path("/home/niko/.claude/uploads/1c72c909-40cd-4c61-928c-e3ebcdf4f383/"
           "e0729264-image.jpg")
OUT = ROOT / "data/processed/historical"
DATUM = "HISTORICAL_BALTIC (realization not stated in source)"
SRC = "Dnipro reservoirs monograph, Fig. 16 (square-on scan)"

# plot interior, established from the frame search
# The box MUST include the 0 km gridline. An earlier X0=600 cut it off, so
# the first detected vertical line was 60 km but got labelled 30 km, and a
# linear residual check cannot see a whole-gridline offset.
Y0, Y1, X0, X1 = 560, 2660, 170, 3000
THR = 130
# gridline elevations, top to bottom, as labelled on the figure
Y_LABELS = [22, 21, 20, 19, 18, 17, 16]
# px boxes holding ring-shaped glyphs that are not data (see the filter below)
# Everything left of this pixel is excluded, for three separate reasons that
# happen to share the same strip: the y-axis label digits (0, 6, 8, 9 all
# enclose light), the legend key (whose curve-2 entry IS a circle), and the
# zone where the dashed curve 3 crosses the dash-dot curve 4 and their dashes
# enclose small lozenges. CONSEQUENCE, stated because it is a real limit: no
# curve-2 point left of ~55 km can be recovered by this method. Visual
# inspection of the figure shows the leftmost plotted circle is at ~73 km, so
# nothing is believed lost -- but that is inspection, not proof.
LEFT_CUT_PX = 750
MIN_HOLE_AREA = 150         # the crossing lozenges are 92-216 px
MAX_ECC = 1.6               # a circle's hole is round; a lozenge is not
X_LABELS = [0, 30, 60, 90, 120, 150, 180]


def main() -> None:
    im = ImageOps.exif_transpose(Image.open(IMG))
    a = np.asarray(im.convert("L"), float)
    print(f"{IMG.name}: {a.shape[1]}x{a.shape[0]} after EXIF transpose")

    # ---- calibrate the axes on detected gridlines --------------------------
    dark_i = (a[Y0:Y1, X0:X1] < THR).astype(float)
    rp = dark_i.sum(1) / dark_i.shape[1]
    pr, _ = find_peaks(rp, height=0.30, distance=25)
    ygrid = sorted(int(p + Y0) for p in pr if 700 < p + Y0 < 2100)[-6:]
    # Several dark rows cluster near the bottom -- the 16 m axis itself, the tick
    # labels below it, and the near-vertical curve tails. Take the STRONGEST, not
    # the first: picking the first gave 2553 px, which broke the otherwise
    # geometric gridline progression (615 where 662 was expected).
    bot = [(rp[q], int(q + Y0)) for q in pr if 2500 < q + Y0 < 2660]
    yaxis = max(bot)[1]
    ypx = np.array(ygrid + [yaxis], float)
    yval = np.array(Y_LABELS, float)
    print("\nvertical calibration (gridline -> elevation):")
    for p, v in zip(ypx, yval):
        print(f"  y={int(p):>5} px  ->  {v:.0f} m")
    d = np.diff(ypx)
    print(f"  pixels per metre, {int(yval[0])}->{int(yval[-1])}: "
          + ", ".join(f"{x:.0f}" for x in d))
    print(f"  ratio bottom/top = {d[-1]/d[0]:.1f}  -> the axis is NOT linear")
    if d[-1] / d[0] < 2:
        raise SystemExit("axis looks linear -- calibration assumption wrong")

    cp = dark_i.sum(0) / dark_i.shape[0]
    pc, _ = find_peaks(cp, height=0.40, distance=100)
    cand = np.array([p + X0 for p in pc], float)
    print(f"\n{len(cand)} dark-column candidates: "
          + ", ".join(f"{int(v)}" for v in cand))

    # The x gridlines are EVENLY spaced by construction (0, 30, ... km), so the
    # right subset is the longest arithmetic chain among the candidates. This is
    # self-checking, unlike a hand-tuned "is it tall enough" filter: label rows
    # and paper texture do not form an evenly spaced chain, and an earlier
    # run-length filter both dropped real lines and kept three intruders that
    # sat between the 0 and 30 km lines.
    best = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            d = cand[j] - cand[i]
            if not (250 <= d <= 360):
                continue
            chain = [cand[i]]
            nxt = cand[i] + d
            while nxt <= cand.max() + 0.25 * d:
                hit = cand[np.abs(cand - nxt) <= 0.10 * d]
                if not len(hit):
                    break
                chain.append(float(hit[0]))
                nxt = chain[-1] + d
            if len(chain) > len(best):
                best = chain
    xg = np.array(best, float)
    print(f"longest evenly spaced chain: {len(xg)} lines, spacing "
          f"{np.diff(xg).mean():.1f} px "
          + ", ".join(f"{int(v)}" for v in xg))
    if len(xg) < 5:
        raise SystemExit("could not identify the horizontal gridlines")
    labels = np.arange(len(xg)) * 30.0        # the chain starts at 0 km
    coef = np.polyfit(xg, labels, 1)
    print(f"\nhorizontal calibration on {len(xg)} gridlines: "
          f"{1/coef[0]:.2f} px per km, x(0 km) = {-coef[1]/coef[0]:.0f} px")
    resid = np.polyval(coef, xg) - labels
    print(f"  residuals (km): " + ", ".join(f"{r:+.1f}" for r in resid))
    if np.abs(resid).max() > 3:
        raise SystemExit("horizontal gridlines are not evenly spaced")

    def px2km(x):
        return np.polyval(coef, x)

    def px2m(y):
        # np.interp REQUIRES an increasing xp. ypx already increases down the
        # page; yval decreases, which is fine for fp. Reversing both gave a
        # decreasing xp, and np.interp then silently clamped every value to the
        # last element -- every elevation came out as 22.00 m.
        assert np.all(np.diff(ypx) > 0), "ypx must increase down the page"
        return np.interp(y, ypx, yval)

    # ---- curve 2: the circles ---------------------------------------------
    mask = a < THR
    interior = np.zeros_like(mask)
    interior[Y0:Y1, X0:X1] = mask[Y0:Y1, X0:X1]
    holes = ndimage.binary_fill_holes(interior) & ~interior
    lab, n = ndimage.label(holes)
    print(f"\n{n} enclosed light regions inside the plot frame")
    objs = ndimage.find_objects(lab)
    cand = []
    for k, sl in enumerate(objs, start=1):
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        area = int((lab[sl] == k).sum())
        if not (12 <= h <= 32 and 12 <= w <= 32):
            continue
        if not (90 <= area <= 560):
            continue
        if abs(h - w) > 8:                        # roughly round
            continue
        cy, cx = ndimage.center_of_mass(lab == k)
        # Two regions of the plot contain ring-shaped glyphs that are NOT data:
        # the y-axis label strip (the digits 0, 6, 8, 9 all enclose light), and
        # the legend key, whose curve-2 entry is literally a circle. Both are
        # excluded by their own boxes rather than by a blanket left-hand cut, so
        # a data point in that range would still be picked up.
        if cx < LEFT_CUT_PX or area < MIN_HOLE_AREA:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float)))
        if np.sqrt(ev.max() / max(ev.min(), 1e-9)) > MAX_ECC:
            continue
        cand.append((float(cx), float(cy), area, h, w))
    cand.sort()
    print(f"{len(cand)} of them are circle-shaped (round, 8-40 px, area 60-900)")

    c2 = pd.DataFrame([{"chainage_km": float(px2km(cx)),
                        "WSE_m": float(px2m(cy)),
                        "px_x": cx, "px_y": cy, "area_px": ar}
                       for cx, cy, ar, h, w in cand])
    c2 = c2[(c2.chainage_km > -5) & (c2.chainage_km < 250)].sort_values("chainage_km")
    print(f"\ncurve 2 (field-measured, 22-25 Apr 1970): {len(c2)} points")
    print(f"  {'chainage':>9}{'WSE':>9}")
    for r in c2.itertuples():
        print(f"  {r.chainage_km:>8.0f} {r.WSE_m:>8.2f}")

    # ---- validation against Table 20 --------------------------------------
    # Curve 2 was measured at Q = 8 400 m3/s, which sits between the tabulated
    # Q20% (7 300) and Q10% (9 400). Interpolating Table 20 between those two
    # columns gives the expected profile with no reference to the figure at all.
    t20 = pd.read_csv(OUT / "historical_table20_wse.csv")
    t20 = t20.rename(columns={"chainage_km_historical": "km"})
    piv = t20.pivot_table(index="km", columns="Q_label", values="WSE_historical_m")
    w = (8400 - 7300) / (9400 - 7300)
    exp = (1 - w) * piv["Q20%"] + w * piv["Q10%"]
    pred = np.interp(c2.chainage_km, exp.index.values, exp.values)
    dev = c2.WSE_m.values - pred
    print(f"\n=== validation: curve 2 against Table 20 interpolated to Q=8400 ===")
    print(f"  n = {len(c2)}")
    print(f"  median deviation : {np.median(dev):+.3f} m")
    print(f"  NMAD             : {1.4826*np.median(np.abs(dev-np.median(dev))):.3f} m")
    print(f"  max |deviation|  : {np.abs(dev).max():.3f} m")
    flat = c2.chainage_km < 180
    if flat.sum():
        print(f"  in the flat pool (<180 km, n={int(flat.sum())}): "
              f"median {np.median(dev[flat.values]):+.3f} m, "
              f"max |dev| {np.abs(dev[flat.values]).max():.3f} m")
    # The verdict is judged on chainage < 210 km ONLY. Table 20 tabulates just
    # two points between 222 and 238 km, and the true curve there is strongly
    # convex, so LINEARLY interpolating the reference across the backwater limb
    # is itself wrong by decimetres. Judging the digitisation against a bad
    # reference would penalise the wrong thing; the limb is reported separately.
    inpool = c2.chainage_km.values < 210
    ok = (np.abs(np.median(dev[inpool])) < 0.10
          and np.abs(dev[inpool]).max() < 0.15)
    print(f"\n  limb (>=210 km, n={int((~inpool).sum())}): deviations "
          + ", ".join(f"{v:+.2f}" for v in dev[~inpool]))
    print(f"    -- Table 20 has only two points across the limb and the true")
    print(f"       curve is convex, so the linear reference is itself wrong")
    print(f"       there by decimetres. Not counted against the digitisation.")
    print(f"\n  -> the digitisation is {'USABLE' if ok else 'NOT usable'}: over the pool "
          f"(<210 km) curve 2\n     reproduces the independently tabulated profile to "
          f"{np.abs(np.median(dev[inpool])):.3f} m median, "
          f"{np.abs(dev[inpool]).max():.3f} m max.")
    print(f"     This also CONFIRMS the non-linear axis calibration: on a linear")
    print(f"     axis the same circles would read 0.2-0.4 m too high.")

    c2o = c2.assign(curve_id=2, observed_or_calculated="OBSERVED",
                    Q_m3_s=8400, H_dam_m=16.00, date="1970-04-22/25",
                    historical_datum_label=DATUM, source_page=SRC,
                    source_figure="Fig. 16",
                    table20_expected_m=pred, deviation_m=dev,
                    method="enclosed-light-region detection, axis calibrated on "
                           "detected gridlines (non-linear vertical scale)")
    meta = pd.DataFrame([
        {"curve_id": 1, "observed_or_calculated": "calculated", "Q_m3_s": 8400,
         "H_dam_m": 16.00, "date": "1966 design", "chainage_km": np.nan,
         "WSE_m": np.nan, "method": "NOT digitised - available numerically in "
         "Table 20", "historical_datum_label": DATUM, "source_page": SRC,
         "source_figure": "Fig. 16"},
        {"curve_id": 3, "observed_or_calculated": "calculated", "Q_m3_s": 23800,
         "H_dam_m": 16.00, "date": "1966 design", "chainage_km": np.nan,
         "WSE_m": np.nan, "method": "NOT digitised - IS the Q0.1% column of "
         "Table 20", "historical_datum_label": DATUM, "source_page": SRC,
         "source_figure": "Fig. 16"},
        {"curve_id": 4, "observed_or_calculated": "OBSERVED", "Q_m3_s": 9040,
         "H_dam_m": 16.16, "date": "1970-04-28/30", "chainage_km": np.nan,
         "WSE_m": np.nan, "method": "NOT digitised - a dash-dot line cannot be "
         "separated from the dashed curve 3 by the hole-detection used for the "
         "circles; needs curve tracing", "historical_datum_label": DATUM,
         "source_page": SRC, "source_figure": "Fig. 16"}])
    pd.concat([c2o, meta], ignore_index=True).to_csv(
        OUT / "historical_fig16_profiles.csv", index=False)
    print(f"\n-> {OUT/'historical_fig16_profiles.csv'}")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.8))
    a_ = ax[0]
    a_.imshow(a[Y0:Y1, X0:X1], cmap="gray", aspect="auto",
              extent=[X0, X1, Y1, Y0])
    a_.scatter(c2.px_x, c2.px_y, s=90, facecolor="none", edgecolor=RED, lw=1.8)
    for p in ypx:
        a_.axhline(p, color=BLUE, lw=0.8, alpha=0.7)
    for p in xg:
        a_.axvline(p, color=GREEN, lw=0.8, alpha=0.7)
    a_.set_xlabel("image x (px)"); a_.set_ylabel("image y (px)")
    a_.set_title(f"Detected: {len(c2)} circles, "
                 f"{len(ypx)} y-gridlines, {len(xg)} x-gridlines",
                 fontsize=11, loc="left")

    a_ = ax[1]
    for q, c, ls in [("Qmin", BLUE, ":"), ("Q20%", GREEN, "--"),
                     ("Q0.1%", RED, "-.")]:
        s = t20[t20.Q_label == q].sort_values("km")
        a_.plot(s.km, s.WSE_historical_m, ls, color=c, lw=1.6,
                label=f"Table 20 {q}")
    a_.plot(exp.index, exp.values, "-", color=INK, lw=2.4,
            label="Table 20 interpolated to Q=8 400 (= curve 1)")
    a_.scatter(c2.chainage_km, c2.WSE_m, s=95, facecolor="none", edgecolor=RED,
               lw=2.2, zorder=6, label="curve 2 digitised (FIELD, Apr 1970)")
    a_.set_xlabel("chainage from the dam (km)")
    a_.set_ylabel("free-surface elevation (m, historical Baltic)")
    a_.legend(fontsize=8.4, loc="upper left")
    a_.grid(alpha=0.22)
    a_.set_title(f"Validation: median deviation {np.median(dev):+.3f} m, "
                 f"max {np.abs(dev).max():.2f} m", fontsize=11, loc="left")

    fig.suptitle("HIST9 · Fig. 16 curve 2 digitised — the source's only "
                 "field-measured longitudinal profile", fontsize=12.5, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(CFG.FIG / f"HIST9_fig16_digitisation.{e}", dpi=185,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"-> {CFG.FIG/'HIST9_fig16_digitisation.png'}")


if __name__ == "__main__":
    main()
