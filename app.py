"""Gradio UI for the filter demo: dropdown -> dynamic sliders -> live preview,
plus error metrics and the error PDF (histogram of result - input).

Sliders are fixed to the normalized range [0, 1] and mapped to each parameter's
real [min, max] in Python. Gradio 6.24.0 raises in preprocess when a slider
value falls outside its current min/max, so reusing sliders with dynamic
min/max across filters lets stale values from the previous filter crash the
request ("Value 9 is greater than maximum value 0.5"). A fixed [0, 1] range
means no value can ever be out of bounds.
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from filters import FILTERS, ERROR_METRICS, entropy

FIRST = list(FILTERS)[0]
MAX_PARAMS = max(len(f["params"]) for f in FILTERS.values())


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


def error_pdf_image(gray, result, width=560, height=320, bins=48):
    """2D bar chart of e = result - gray, rendered in pure numpy, labels via PIL."""
    e = (result.astype(np.float64) - gray.astype(np.float64)).ravel()
    lo, hi = float(e.min()), float(e.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    counts, _ = np.histogram(e, bins=bins, range=(lo, hi))
    hmax = counts.max() or 1

    top, bottom, side = 16, 34, 8        # margins reserving room for labels
    plot_h = height - top - bottom
    plot_w = width - 2 * side
    canvas = np.full((height, width, 3), (14, 16, 22), dtype=np.uint8)
    blue = np.array((86, 168, 255), dtype=np.uint8)

    bw = plot_w / bins
    base = height - bottom
    for i, c in enumerate(counts):
        if c == 0:
            continue
        h = max(2, int(round(c / hmax * (plot_h - 2))))
        x0 = side + int(i * bw)
        x1 = side + int((i + 1) * bw) - 1
        canvas[base - h:base, x0:x1 + 1] = blue
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
    name = "error = result - input"
    tw2 = d.textlength(name, font=font)
    d.text((int((width - tw2) / 2), base + 18), name, font=font, fill=(210, 220, 240))
    return np.asarray(pil), (lo, hi, float(e.mean()), float(e.std()))


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


def render(img, name, vals):
    gray, result = apply(img, name, vals)
    if result is None:
        return None, None, None
    pdf, rng = error_pdf_image(gray, result)
    return result, pdf, metrics_markdown(gray, result, rng)


# ---------------------------------------------------------------------------
# event handlers
# ---------------------------------------------------------------------------

def defaults_for(name):
    params = FILTERS[name]["params"]
    return [_norm(p, p["default"]) for p in params] + [0.0] * (MAX_PARAMS - len(params))


def choose(img, name):
    """Dropdown changed: reset state to this filter's defaults, show its sliders."""
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
    vals = defaults_for(name)
    result, pdf, md = render(img, name, vals)
    return name, *updates, vals, f"$$ {FILTERS[name]['formula']} $$", result, pdf, md, _params_md(name, vals)


def make_adjust(i):
    """One handler per slider: only this slider's own (normalized) value feeds the filter."""
    def on_change(img, name, vals, v):
        vals = list(vals) if vals else defaults_for(name)
        vals[i] = v
        result, pdf, md = render(img, name, vals)
        return result, pdf, md, _params_md(name, vals), vals
    return on_change


def on_img(img, name, vals):
    """New image: re-run the current filter with the current slider values."""
    if not vals:
        vals = defaults_for(name)
    result, pdf, md = render(img, name, vals)
    return result, pdf, md, _params_md(name, vals)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Image Filter Demo") as demo:
    gr.Markdown("# Image Filter Demo\nFilters from Gonzalez & Woods, *Digital Image Processing* (4th ed.)")
    name = gr.Dropdown(list(FILTERS), label="Filter", value=FIRST)
    formula = gr.Markdown(f"$$ {FILTERS[FIRST]['formula']} $$")
    sliders = [gr.Slider(0, 1, value=0, visible=False) for _ in range(MAX_PARAMS)]
    params = gr.Markdown("")
    vals = gr.State(None)
    with gr.Row():
        inp = gr.Image(label="Input image", type="numpy")
        out = gr.Image(label="Result", type="numpy", format="png")
    with gr.Row():
        pdf = gr.Image(label="Error PDF (histogram of result − input)", type="numpy", format="png")
        metrics = gr.Markdown("")
    current = gr.State(FIRST)

    name.change(choose, [inp, name], [current, *sliders, vals, formula, out, pdf, metrics, params])
    for i, s in enumerate(sliders):
        s.change(make_adjust(i), [inp, current, vals, s], [out, pdf, metrics, params, vals])
    inp.change(on_img, [inp, current, vals], [out, pdf, metrics, params])
    demo.load(choose, [inp, name], [current, *sliders, vals, formula, out, pdf, metrics, params])

if __name__ == "__main__":
    demo.launch()
