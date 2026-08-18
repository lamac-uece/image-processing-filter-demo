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

from filters import (
    CAPTIONS, FILTERS, ERROR_METRICS, PRESETS, entropy,
    me, rmse, mae, nmse, psnr, snr, correlation, jaccard,
)

FIRST = list(FILTERS)[0]
MAX_PARAMS = max(len(f["params"]) for f in FILTERS.values())
NONE = "— (none) —"

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
    if name == NONE:
        return ""
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


def apply_stack(img, stack):
    """Apply an ordered stack of filters: RGB numpy in -> (gray uint8, result uint8)."""
    if img is None:
        return None, None
    gray = _gray(img)
    result = gray
    for entry in stack or []:
        params = FILTERS[entry["name"]]["params"]
        kwargs = {p["name"]: _actual(p, entry["vals"][i]) for i, p in enumerate(params)}
        result = FILTERS[entry["name"]]["fn"](result, **kwargs)
    return gray, result


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


def verdict(psnr):
    """Color-coded quality verdict from PSNR (dB)."""
    if psnr == float("inf"):
        return "🟢 Perfect (identical)"
    if psnr >= 40:
        return "🟢 Excellent"
    if psnr >= 30:
        return "🟢 Good"
    if psnr >= 25:
        return "🟡 Fair"
    if psnr >= 20:
        return "🟠 Poor"
    return "🔴 Very poor"


_METRIC_ROWS = [  # (label, fn, reference, family) — bar = value/reference, clipped
    ("ME",        me,   255.0, "err"),
    ("RMSE",      rmse, 255.0, "err"),
    ("MAE",       mae,  255.0, "err"),
    ("NMSE",      nmse, 1.0,   "err"),
    ("error mean", None, 255.0, "err"),
    ("error std", None, 255.0, "err"),
    ("PSNR",      psnr, 50.0,  "qual"),
    ("SNR",       snr,  50.0,  "qual"),
    ("Corr",      correlation, 1.0, "qual"),
    ("Jaccard",   jaccard, 1.0, "qual"),
]


def metrics_bars(gray, result, rng, width=560, height=320):
    """Horizontal bar chart of the error/quality metrics, each normalized to a
    fixed reference so magnitudes are comparable across filters."""
    lo, hi, mean, std = rng
    values = {}
    for label, fn, ref, _family in _METRIC_ROWS:
        v = fn(gray, result) if fn is not None else (abs(mean) if label == "error mean" else std)
        values[label] = v
    colors = {"err": (240, 140, 90), "qual": (120, 210, 140)}

    top, bottom, side = 12, 12, 8
    rows = len(_METRIC_ROWS)
    row_h = (height - top - bottom) / rows
    label_w, value_w, bar_x0 = 100, 46, 108
    bar_x1 = width - side - value_w
    canvas = np.full((height, width, 3), (14, 16, 22), dtype=np.uint8)

    for i, (label, _f, ref, family) in enumerate(_METRIC_ROWS):
        v = values[label]
        if v == float("inf") or np.isnan(v):
            frac = 1.0
        else:
            frac = min(v / ref, 1.0)
        y0 = int(top + i * row_h)
        y1 = int(top + (i + 1) * row_h) - 1
        ym = (y0 + y1) // 2
        color = np.array(colors[family], dtype=np.uint8)
        if frac > 0:
            x1 = int(bar_x0 + frac * (bar_x1 - bar_x0))
            canvas[y0:y1 + 1, bar_x0:x1] = color
        # baseline tick at full scale
        canvas[y0:y1 + 1, bar_x1] = (110, 120, 140)

    # labels + values via PIL
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=12)
    small = ImageFont.load_default(size=10)
    grey = (210, 220, 240)
    for i, (label, _f, ref, _family) in enumerate(_METRIC_ROWS):
        y = int(top + i * row_h + row_h / 2 - 7)
        d.text((side, y), label, font=small, fill=grey)
        v = values[label]
        txt = "∞" if v == float("inf") else (f"{v:.2f}" if v < 100 else f"{v:.0f}")
        tw = d.textlength(txt, font=small)
        d.text((bar_x1 - tw - 4, y), txt, font=small, fill=(190, 200, 220))
    return np.asarray(pil)


def metrics_markdown(gray, result, rng):
    lo, hi, mean, std = rng
    rows = [f"**Verdict: {verdict(psnr(gray, result))}**", "", "| measure | value |", "|---|---|"]
    for label, fn, _ in ERROR_METRICS:
        rows.append(f"| {label} | {_fmt(fn(gray, result))} |")
    rows.append(f"| error mean | {mean:.4f} |")
    rows.append(f"| error std | {std:.4f} |")
    rows.append(f"| error range | [{lo:.2f}, {hi:.2f}] |")
    rows.append(f"| entropy (input) | {entropy(gray):.4f} |")
    rows.append(f"| entropy (result) | {entropy(result):.4f} |")
    return "\n".join(rows)


def _stack_to_qp(stack):
    """Query-string fields that encode an ordered filter stack."""
    q = {"stack": str(len(stack or []))}
    for i, entry in enumerate(stack or []):
        q[f"s{i}"] = entry["name"]
        for j in range(len(FILTERS[entry["name"]]["params"])):
            q[f"s{i}p{j}"] = f"{float(entry['vals'][j]):.3f}"
    return q


def _stack_md(stack):
    """Render the ordered filter stack as Markdown."""
    stack = stack or []
    if not stack:
        return "**Stack:** _(empty — the current filter previews below)_"
    n = len(stack)
    lines = [f"**Stack ({n} filter{'s' if n != 1 else ''}, applied in order):**", ""]
    for i, entry in enumerate(stack, 1):
        params = FILTERS[entry["name"]]["params"]
        if params:
            parts = [f"{p['name']}={_actual(p, entry['vals'][j]):g}"
                     for j, p in enumerate(params)]
            detail = ", ".join(parts)
        else:
            detail = "no params"
        lines.append(f"{i}. **{entry['name']}** — {detail}")
    return "\n".join(lines)


def share_md(name, vals, stack, base=None):
    """Deep link query string (or full URL when base host is known)."""
    q = {}
    if name != NONE:
        q["filter"] = name
        q.update({f"p{i}": f"{float(vals[i]):.3f}" for i in range(len(FILTERS[name]["params"]))})
    q.update(_stack_to_qp(stack))
    full = f"{base}/?{urlencode(q)}" if base else f"?{urlencode(q)}"
    return f"**Deep link:** `{full}`"


def render_all(img, name, vals, base=None, stack=None):
    """Apply the active pipeline = committed stack with the current filter on top,
    then build every output panel. Returns 9 values."""
    if name != NONE and not vals:
        vals = defaults_for(name)
    chain = list(stack or [])
    if name != NONE:
        chain.append({"name": name, "vals": vals})
    gray, result = apply_stack(img, chain)
    if result is None:
        return (None,) * 9
    pdf, rng = error_pdf_image(gray, result)
    return (
        result,
        hist_image(gray, (170, 180, 205)),
        hist_image(result, (86, 168, 255)),
        diff_map(gray, result),
        pdf,
        metrics_bars(gray, result, rng),
        metrics_markdown(gray, result, rng),
        _params_md(name, vals),
        share_md(name, vals, stack, base),
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


def choose(img, name, stack=None, base=None):
    """Dropdown changed: make `name` the current (top) filter and re-render the
    live chain = committed stack + current filter."""
    if name == NONE:
        vals = []
        slider_updates = [gr.update(visible=False)] * MAX_PARAMS
        formula = "*(no filter selected — showing the committed stack)*"
        caption = "Pick a filter to stack it on top of the current result."
        presets_update = gr.update(visible=False)
    else:
        vals = defaults_for(name)
        slider_updates = _slider_updates(name)
        formula = f"$$ {FILTERS[name]['formula']} $$"
        caption = f"💡 {CAPTIONS[name]}"
        presets_update = _presets_update(name)
    return (
        name, *slider_updates, vals,
        formula, caption, presets_update,
        *render_all(img, name, vals, base, stack),
    )


def make_adjust(i):
    """One handler per slider: only this slider's own (normalized) value feeds the filter."""
    def on_change(img, name, vals, v, stack=None):
        if name == NONE:
            return *render_all(img, NONE, [], None, stack), []
        vals = list(vals) if vals else defaults_for(name)
        vals[i] = v
        return *render_all(img, name, vals, None, stack), vals
    return on_change


def on_img(img, name, vals, stack=None):
    """New image: re-run the active pipeline with the current slider values."""
    if not vals and name != NONE:
        vals = defaults_for(name)
    return render_all(img, name, vals, None, stack)


def on_preset(img, name, vals, label, stack=None):
    """Apply a preset: set sliders from PRESETS[name], re-render."""
    if name == NONE:
        return (*[gr.update()] * MAX_PARAMS, [], *render_all(img, NONE, [], None, stack))
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
    return *slider_updates, vals, *render_all(img, name, vals, None, stack)


def on_reset(img, name, stack=None):
    """Reset to this filter's defaults (stack is preserved)."""
    return choose(img, name, stack, None)


def _stack_render(img, name, vals, stack, base=None):
    """Re-render after a stack mutation. Returns the stack-mutation output tuple."""
    return (*render_all(img, name, vals, base, stack), stack, _stack_md(stack))


def on_add(img, name, vals, stack):
    """Commit the current filter into the stack, then clear the current filter."""
    stack = list(stack or [])
    if name != NONE:
        vals = list(vals) if vals else defaults_for(name)
        stack.append({"name": name, "vals": vals[:]})
    dd = [gr.update(value=NONE) for _ in CHAPTERS]
    sliders_upd = [gr.update(visible=False)] * MAX_PARAMS
    return (*dd, NONE, *sliders_upd, [],
            "*(no filter selected — showing the committed stack)*",
            "Pick a filter to stack it on top of the current result.",
            gr.update(visible=False),
            *render_all(img, NONE, [], None, stack),
            stack, _stack_md(stack))


def on_pop(img, name, vals, stack):
    """Drop the last committed filter from the stack."""
    vals = list(vals) if vals else (defaults_for(name) if name != NONE else [])
    stack = list(stack or [])
    if stack:
        stack.pop()
    return _stack_render(img, name, vals, stack)


def on_clear(img, name, vals, stack):
    """Empty the committed stack (the current filter, if any, stays on top)."""
    vals = list(vals) if vals else (defaults_for(name) if name != NONE else [])
    return _stack_render(img, name, vals, [])


def make_tab_select(chapter):
    """Tab clicked: sync this chapter's dropdown to the current filter (or none)."""
    def on_tab(img, current, vals, stack):
        value = current if current in CH_FILTERS[chapter] else NONE
        return (gr.update(value=value), current, *[gr.update()] * MAX_PARAMS,
                vals, *[gr.update()] * 12)
    return on_tab


def _parse_deep_link(qp):
    """(?filter=...&p0=...&stack=n&s0=...&s0p0=...) -> (name, vals, stack)."""
    raw = qp.get("filter")
    name = raw if raw in FILTERS else NONE
    vals = defaults_for(name) if name != NONE else []
    if name != NONE:
        for i in range(MAX_PARAMS):
            if f"p{i}" in qp:
                try:
                    vals[i] = min(max(float(qp[f"p{i}"]), 0.0), 1.0)
                except ValueError:
                    pass
    stack = []
    try:
        n = int(qp.get("stack", "0"))
    except (TypeError, ValueError):
        n = 0
    for i in range(n):
        sname = qp.get(f"s{i}")
        if sname not in FILTERS:
            continue
        svals = defaults_for(sname)
        for j in range(MAX_PARAMS):
            key = f"s{i}p{j}"
            if key in qp:
                try:
                    svals[j] = min(max(float(qp[key]), 0.0), 1.0)
                except ValueError:
                    pass
        stack.append({"name": sname, "vals": svals})
    return name, vals, stack


def _dd_updates(name):
    """Each chapter dropdown shows `name` if it belongs there, else none."""
    return [gr.update(value=(name if name in CH_FILTERS[ch] else NONE))
            for ch, _ in CHAPTERS]


def on_load(request: gr.Request, img):
    """Page load: restore state from URL query params (deep link), else defaults."""
    base = None
    qp = dict(request.query_params) if request else {}
    if request is not None and request.headers:
        host = request.headers.get("host")
        if host:
            base = f"http://{host}"
    name, vals, stack = _parse_deep_link(qp)
    if name == NONE and not stack:
        name = FIRST
        vals = defaults_for(name)
    dd_updates = _dd_updates(name)
    return *choose(img, name, stack, base), stack, _stack_md(stack), *dd_updates


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
                dropdowns[ch] = gr.Dropdown([NONE, *CH_FILTERS[ch]], label="Filter", value=CH_FILTERS[ch][0])
    formula = gr.Markdown(f"$$ {FILTERS[FIRST]['formula']} $$")
    caption = gr.Markdown(f"💡 {CAPTIONS[FIRST]}")
    presets = gr.Dropdown([], label="Try a preset", visible=False)
    sliders = [gr.Slider(0, 1, value=0, visible=False) for _ in range(MAX_PARAMS)]
    params = gr.Markdown("")
    gr.Markdown(
        "### Filter stack\n"
        "Pick a filter and tune its sliders — the **Result** previews it applied "
        "on top of the stack. Press **＋ Add** to commit it."
    )
    with gr.Row():
        add_btn = gr.Button("＋ Add to stack", variant="primary")
        pop_btn = gr.Button("↩ Remove last")
        clear_btn = gr.Button("🗑 Clear stack")
    stack_view = gr.Markdown("**Stack:** _(empty — the current filter previews below)_")
    with gr.Row():
        reset = gr.Button("↺ Reset to defaults")
        share = gr.Markdown("")
    vals = gr.State(None)
    stack = gr.State([])
    with gr.Row():
        inp = gr.Image(label="Input image", type="numpy")
        out = gr.Image(label="Result", type="numpy", format="png")
    with gr.Row():
        hist_in = gr.Image(label="Input histogram", type="numpy", format="png")
        hist_out = gr.Image(label="Result histogram", type="numpy", format="png")
    with gr.Row():
        pdf = gr.Image(label="Error PDF (histogram of result − input)", type="numpy", format="png")
        bars = gr.Image(label="Error metrics (bars: value / reference)", type="numpy", format="png")
    with gr.Row():
        diff = gr.Image(label="Difference |result − input|", type="numpy", format="png")
        metrics = gr.Markdown("")
    current = gr.State(FIRST)

    # output lists (must match each handler's return order)
    CHOOSE_OUT = [current, *sliders, vals, formula, caption, presets,
                  out, hist_in, hist_out, diff, pdf, bars, metrics, params, share]
    RENDER_OUT = [out, hist_in, hist_out, diff, pdf, bars, metrics, params, share]
    PRESET_OUT = [*sliders, vals, *RENDER_OUT]
    STACK_OUT = [*RENDER_OUT, stack, stack_view]
    ADD_OUT = [*dropdowns.values(), current, *sliders, vals, formula, caption, presets,
               *RENDER_OUT, stack, stack_view]
    LOAD_OUT = [*CHOOSE_OUT, stack, stack_view, *dropdowns.values()]

    for ch, dd in dropdowns.items():
        dd.change(choose, [inp, dd, stack], CHOOSE_OUT)
        tab_out = [dd, current, *sliders, vals, formula, caption, presets,
                   out, hist_in, hist_out, diff, pdf, bars, metrics, params, share]
        tabs[ch].select(make_tab_select(ch), [inp, current, vals, stack], tab_out)
    for i, s in enumerate(sliders):
        s.change(make_adjust(i), [inp, current, vals, s, stack], [*RENDER_OUT, vals])
    presets.change(on_preset, [inp, current, vals, presets, stack], PRESET_OUT)
    add_btn.click(on_add, [inp, current, vals, stack], ADD_OUT)
    pop_btn.click(on_pop, [inp, current, vals, stack], STACK_OUT)
    clear_btn.click(on_clear, [inp, current, vals, stack], STACK_OUT)
    reset.click(on_reset, [inp, current, stack], CHOOSE_OUT)
    inp.change(on_img, [inp, current, vals, stack], RENDER_OUT)
    demo.load(on_load, [inp], LOAD_OUT)

if __name__ == "__main__":
    demo.launch()
