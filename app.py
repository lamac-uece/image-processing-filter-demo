"""Gradio UI for the filter demo: chapter tabs -> dropdown -> dynamic sliders ->
live preview, plus error metrics, error PDF, intensity histograms and a spatial
difference map. Classroom extras: plain-language captions, parameter presets, a
reset button and deep links (?filter=...&p0=...) restored via gr.Request.

Sliders are fixed to the normalized range [0, 1] and mapped to each parameter's
real [min, max] in Python. Gradio 6.24.0 raises in preprocess when a slider
value falls outside its current min/max, so reusing sliders with dynamic
min/max across filters lets stale values from the previous filter crash the
request ("Value 9 is greater than maximum value 0.5"). A fixed [0, 1] range
means no value can ever be out of bounds.
"""

from urllib.parse import urlencode

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from filters import CAPTIONS, FILTERS, ERROR_METRICS, PRESETS, entropy

FIRST = list(FILTERS)[0]
MAX_PARAMS = max(len(f["params"]) for f in FILTERS.values())

CHAPTERS = [
    ("Ch2", "Ch2 · Basics"),
    ("Ch3", "Ch3 · Intensity & Spatial"),
    ("Ch4", "Ch4 · Frequency Domain"),
    ("Ch5", "Ch5 · Noise & Restoration"),
]
CH_FILTERS = {ch: [n for n in FILTERS if n.startswith(ch + " ·")] for ch, _ in CHAPTERS}


# ---------------------------------------------------------------------------
# param mapping: slider is always [0, 1] -> real value snapped to the step grid
# ---------------------------------------------------------------------------

def _norm(p, v):
    """real param value -> [0, 1] slider position."""
    return (v - p["min"]) / (p["max"] - p["min"])


def _actual(p, v):
    """[0, 1] slider position -> real param value, snapped to p['step']."""
    step = p.get("step", 1.0)
    v = min(max(float(v), 0.0), 1.0)
    a = p["min"] + v * (p["max"] - p["min"])
    a = p["min"] + round((a - p["min"]) / step) * step
    a = min(max(a, p["min"]), p["max"])
    return int(a) if p.get("int") else a


def _params_md(name, values):
    params = FILTERS[name]["params"]
    if not params:
        return ""
    parts = [f"{p['name']} = {_actual(p, values[i]):g}" for i, p in enumerate(params)]
    return "**Params:** " + ", ".join(parts)


# ---------------------------------------------------------------------------
# core: apply filter, then measure error between input and result
# ---------------------------------------------------------------------------

def _gray(img):
    return np.asarray(Image.fromarray(img).convert("L"))


def apply(img, name, values):
    """RGB numpy in -> (gray uint8, result uint8), or (None, None)."""
    if img is None:
        return None, None
    gray = _gray(img)
    params = FILTERS[name]["params"]
    kwargs = {p["name"]: _actual(p, values[i]) for i, p in enumerate(params)}
    return gray, FILTERS[name]["fn"](gray, **kwargs)


# ---------------------------------------------------------------------------
# chart renderers (pure numpy + PIL labels, no matplotlib)
# ---------------------------------------------------------------------------

def _bar_chart(counts, lo, hi, x_label, color, width, height):
    """2D labeled bar chart of a histogram. Returns uint8 (H, W, 3)."""
    hmax = counts.max() or 1
    top, bottom, side = 16, 34, 8        # margins reserving room for labels
    plot_h = height - top - bottom
    plot_w = width - 2 * side
    canvas = np.full((height, width, 3), (14, 16, 22), dtype=np.uint8)
    c = np.array(color, dtype=np.uint8)

    bw = plot_w / len(counts)
    base = height - bottom
    for i, n in enumerate(counts):
        if n == 0:
            continue
        h = max(2, int(round(n / hmax * (plot_h - 2))))
        x0 = side + int(i * bw)
        x1 = side + int((i + 1) * bw) - 1
        canvas[base - h:base, x0:x1 + 1] = c
    canvas[base - 1, side:width - side] = (110, 120, 140)   # baseline

    # axis labels
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=12)
    small = ImageFont.load_default(size=10)
    grey = (190, 200, 220)
    d.text((side, top - 14), f"count (max {hmax})", font=small, fill=grey)
    d.text((side, base + 4), f"{lo:.0f}", font=small, fill=grey)
    tw = d.textlength(f"{hi:.0f}", font=small)
    d.text((width - side - tw, base + 4), f"{hi:.0f}", font=small, fill=grey)
    tw2 = d.textlength(x_label, font=font)
    d.text((int((width - tw2) / 2), base + 18), x_label, font=font, fill=(210, 220, 240))
    return np.asarray(pil)


def error_pdf_image(gray, result, width=560, height=320, bins=48):
    """Bar chart of e = result - gray. Returns (img, (lo, hi, mean, std))."""
    e = (result.astype(np.float64) - gray.astype(np.float64)).ravel()
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    counts, _ = np.histogram(e, bins=bins, range=(lo, hi))
    return _bar_chart(counts, lo, hi, "error = result - input", (86, 168, 255), width, height), \
        (lo, hi, float(e.mean()), float(e.std()))


def hist_image(img, color, width=280, height=180, bins=64):
    """Gray-level histogram (0..255) of an image."""
    counts, _ = np.histogram(np.asarray(img).ravel(), bins=bins, range=(0, 256))
    return _bar_chart(counts, 0, 255, "gray level", color, width, height)


def _heat_lut():
    """Inferno-like 256x3 RGB lookup table built from anchors."""
    anchors = np.array([
        [0, 0, 4], [7, 0, 50], [39, 1, 119], [91, 35, 158],
        [150, 65, 185], [216, 118, 164], [254, 180, 113], [255, 230, 60],
    ], dtype=float)
    xs = np.linspace(0, 255, len(anchors))
    lut = np.stack([np.interp(np.arange(256), xs, anchors[:, c]) for c in range(3)], axis=1)
    return lut.astype(np.uint8)


_HEAT = _heat_lut()


def diff_map(gray, result, width=280, height=280):
    """|result - gray| as a heatmap, downscaled to a fixed preview size."""
    d = np.abs(result.astype(np.float64) - gray.astype(np.float64))
    mx = d.max()
    idx = (d * 255 / mx).astype(np.uint8) if mx > 0 else np.zeros_like(d, dtype=np.uint8)
    small = Image.fromarray(_HEAT[idx]).resize((width, height), Image.BILINEAR)
    return np.asarray(small)


def _fmt(v):
    return "∞" if v == float("inf") else f"{v:.4f}"


def metrics_markdown(gray, result, rng):
    lo, hi, mean, std = rng
    rows = ["| measure | value |", "|---|---|"]
    for label, fn, _ in ERROR_METRICS:
        rows.append(f"| {label} | {_fmt(fn(gray, result))} |")
    rows.append(f"| error mean | {mean:.4f} |")
    rows.append(f"| error std | {std:.4f} |")
    rows.append(f"| error range | [{lo:.2f}, {hi:.2f}] |")
    rows.append(f"| entropy (input) | {entropy(gray):.4f} |")
    rows.append(f"| entropy (result) | {entropy(result):.4f} |")
    return "\n".join(rows)


def share_md(name, vals, base=None):
    """Deep link query string (or full URL when base host is known)."""
    q = urlencode({"filter": name, **{
        f"p{i}": f"{float(vals[i]):.3f}" for i in range(len(FILTERS[name]["params"]))
    }})
    full = f"{base}/?{q}" if base else f"?{q}"
    return f"**Deep link:** `{full}`"


def render_all(img, name, vals, base=None):
    """Apply filter and build every output panel. Returns 8 values."""
    gray, result = apply(img, name, vals)
    if result is None:
        return (None,) * 8
    pdf, rng = error_pdf_image(gray, result)
    return (
        result,
        hist_image(gray, (170, 180, 205)),
        hist_image(result, (86, 168, 255)),
        diff_map(gray, result),
        pdf,
        metrics_markdown(gray, result, rng),
        _params_md(name, vals),
        share_md(name, vals, base),
    )


def defaults_for(name):
    params = FILTERS[name]["params"]
    return [_norm(p, p["default"]) for p in params] + [0.0] * (MAX_PARAMS - len(params))


# ---------------------------------------------------------------------------
# event handlers
# ---------------------------------------------------------------------------

def _slider_updates(name):
    params = FILTERS[name]["params"]
    updates = []
    for i in range(MAX_PARAMS):
        if i < len(params):
            p = params[i]
            updates.append(gr.update(
                visible=True,
                label=f"{p['name']} ({p['min']}–{p['max']})",
                value=_norm(p, p["default"]),
                step=p.get("step", 1.0) / (p["max"] - p["min"]),
            ))
        else:
            updates.append(gr.update(visible=False))
    return updates


def _presets_update(name):
    presets = PRESETS.get(name, [])
    return gr.update(choices=[p["label"] for p in presets], value=None, visible=bool(presets))


def choose(img, name, base=None):
    """Dropdown changed: reset state to this filter's defaults, show its sliders."""
    vals = defaults_for(name)
    return (
        name, *_slider_updates(name), vals,
        f"$$ {FILTERS[name]['formula']} $$", f"💡 {CAPTIONS[name]}", _presets_update(name),
        *render_all(img, name, vals, base),
    )


def make_adjust(i):
    """One handler per slider: only this slider's own (normalized) value feeds the filter."""
    def on_change(img, name, vals, v, base=None):
        vals = list(vals) if vals else defaults_for(name)
        vals[i] = v
        return *render_all(img, name, vals, base), vals
    return on_change


def on_img(img, name, vals, base=None):
    """New image: re-run the current filter with the current slider values."""
    if not vals:
        vals = defaults_for(name)
    return render_all(img, name, vals, base)


def on_preset(img, name, vals, label, base=None):
    """Apply a preset: set sliders from PRESETS[name], re-render."""
    vals = list(vals) if vals else defaults_for(name)
    for pr in PRESETS.get(name, []):
        if pr["label"] == label:
            for i, p in enumerate(FILTERS[name]["params"]):
                if p["name"] in pr["values"]:
                    vals[i] = _norm(p, pr["values"][p["name"]])
            break
    slider_updates = [
        gr.update(value=vals[i]) if i < len(FILTERS[name]["params"]) else gr.update()
        for i in range(MAX_PARAMS)
    ]
    return *slider_updates, vals, *render_all(img, name, vals, base)


def on_reset(img, name, base=None):
    """Reset to this filter's defaults."""
    return choose(img, name, base)


def make_tab_select(chapter):
    """Tab clicked: jump to the chapter's first filter, or sync if current is here."""
    def on_tab(img, current, vals):
        if current in CH_FILTERS[chapter]:
            return (gr.update(value=current), current, *[gr.update()] * MAX_PARAMS,
                    vals, *[gr.update()] * 11)
        name = CH_FILTERS[chapter][0]
        return gr.update(value=name), *choose(img, name)
    return on_tab


def _parse_deep_link(qp):
    """(?filter=...&p0=...&p1=...) -> (name, vals) or (None, None)."""
    name = qp.get("filter")
    if name not in FILTERS:
        return None, None
    vals = defaults_for(name)
    for i in range(MAX_PARAMS):
        if f"p{i}" in qp:
            try:
                vals[i] = min(max(float(qp[f"p{i}"]), 0.0), 1.0)
            except ValueError:
                pass
    return name, vals


def on_load(request: gr.Request, img, base=None):
    """Page load: restore state from URL query params (deep link), else defaults."""
    qp = dict(request.query_params) if request else {}
    if request is not None and request.headers:
        host = request.headers.get("host")
        if host:
            base = f"http://{host}"
    name, vals = _parse_deep_link(qp)
    if name is None:
        name = FIRST
        vals = defaults_for(name)
    dd_updates = [
        gr.update(value=name if name in CH_FILTERS[ch] else CH_FILTERS[ch][0])
        for ch, _ in CHAPTERS
    ]
    return *choose(img, name, base), *dd_updates


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Image Filter Demo") as demo:
    gr.Markdown("# Image Filter Demo\nFilters from Gonzalez & Woods, *Digital Image Processing* (4th ed.)")
    dropdowns = {}
    tabs = {}
    with gr.Tabs():
        for ch, label in CHAPTERS:
            with gr.Tab(label) as tab:
                tabs[ch] = tab
                dropdowns[ch] = gr.Dropdown(CH_FILTERS[ch], label="Filter", value=CH_FILTERS[ch][0])
    formula = gr.Markdown(f"$$ {FILTERS[FIRST]['formula']} $$")
    caption = gr.Markdown(f"💡 {CAPTIONS[FIRST]}")
    presets = gr.Dropdown([], label="Try a preset", visible=False)
    sliders = [gr.Slider(0, 1, value=0, visible=False) for _ in range(MAX_PARAMS)]
    params = gr.Markdown("")
    with gr.Row():
        reset = gr.Button("↺ Reset to defaults")
        share = gr.Markdown("")
    vals = gr.State(None)
    with gr.Row():
        inp = gr.Image(label="Input image", type="numpy")
        out = gr.Image(label="Result", type="numpy", format="png")
    with gr.Row():
        hist_in = gr.Image(label="Input histogram", type="numpy", format="png")
        hist_out = gr.Image(label="Result histogram", type="numpy", format="png")
    with gr.Row():
        pdf = gr.Image(label="Error PDF (histogram of result − input)", type="numpy", format="png")
        diff = gr.Image(label="Difference |result − input|", type="numpy", format="png")
    metrics = gr.Markdown("")
    current = gr.State(FIRST)

    # output lists (must match each handler's return order)
    CHOOSE_OUT = [current, *sliders, vals, formula, caption, presets,
                  out, hist_in, hist_out, diff, pdf, metrics, params, share]
    RENDER_OUT = [out, hist_in, hist_out, diff, pdf, metrics, params, share]
    PRESET_OUT = [*sliders, vals, *RENDER_OUT]
    LOAD_OUT = [*CHOOSE_OUT, *dropdowns.values()]

    for ch, dd in dropdowns.items():
        dd.change(choose, [inp, dd], CHOOSE_OUT)
        tab_out = [dd, current, *sliders, vals, formula, caption, presets,
                   out, hist_in, hist_out, diff, pdf, metrics, params, share]
        tabs[ch].select(make_tab_select(ch), [inp, current, vals], tab_out)
    for i, s in enumerate(sliders):
        s.change(make_adjust(i), [inp, current, vals, s], [*RENDER_OUT, vals])
    presets.change(on_preset, [inp, current, vals, presets], PRESET_OUT)
    reset.click(on_reset, [inp, current], CHOOSE_OUT)
    inp.change(on_img, [inp, current, vals], RENDER_OUT)
    demo.load(on_load, [inp], LOAD_OUT)

if __name__ == "__main__":
    demo.launch()
