"""Gradio UI for the filter demo: dropdown -> dynamic sliders -> live preview.

Run: python app.py
"""

import gradio as gr
import numpy as np
from PIL import Image

from filters import FILTERS

FIRST = list(FILTERS)[0]
MAX_PARAMS = max(len(f["params"]) for f in FILTERS.values())


def run_filter(img, name, values):
    if img is None:
        return None
    gray = np.asarray(Image.fromarray(img).convert("L"))
    params = FILTERS[name]["params"]
    kwargs = {p["name"]: values[i] for i, p in enumerate(params)}
    return FILTERS[name]["fn"](gray, **kwargs)


def defaults_for(name):
    params = FILTERS[name]["params"]
    return [p["default"] for p in params] + [0] * (MAX_PARAMS - len(params))


def choose(img, name):
    """Dropdown changed: reset state to this filter's defaults, show its sliders."""
    params = FILTERS[name]["params"]
    updates = []
    for i in range(MAX_PARAMS):
        if i < len(params):
            p = params[i]
            u = dict(visible=True, label=p["name"], minimum=p["min"],
                     maximum=p["max"], step=p.get("step", 0.1), value=p["default"])
            if p.get("int"):
                u["precision"] = 0
            updates.append(gr.update(**u))
        else:
            updates.append(gr.update(visible=False))
    vals = defaults_for(name)
    return name, *updates, vals, f"$$ {FILTERS[name]['formula']} $$", run_filter(img, name, vals)


def make_adjust(i):
    """One handler per slider: only this slider's own (in-bounds) value feeds the filter."""
    def on_change(img, name, vals, v):
        vals = list(vals) if vals else defaults_for(name)
        vals[i] = v
        return run_filter(img, name, vals), vals
    return on_change


def on_img(img, name, vals):
    """New image: re-run the current filter with the current slider values."""
    if not vals:
        vals = defaults_for(name)
    return run_filter(img, name, vals)


with gr.Blocks(title="Image Filter Demo") as demo:
    gr.Markdown("# Image Filter Demo\nFilters from Gonzalez & Woods, *Digital Image Processing* (4th ed.)")
    name = gr.Dropdown(list(FILTERS), label="Filter", value=FIRST)
    formula = gr.Markdown(f"$$ {FILTERS[FIRST]['formula']} $$")
    sliders = [gr.Slider(0, 1, value=0, visible=False) for _ in range(MAX_PARAMS)]
    vals = gr.State(None)
    with gr.Row():
        inp = gr.Image(label="Input image", type="numpy")
        out = gr.Image(label="Result", type="numpy")
    current = gr.State(FIRST)

    name.change(choose, [inp, name], [current, *sliders, vals, formula, out])
    for i, s in enumerate(sliders):
        s.change(make_adjust(i), [inp, current, vals, s], [out, vals])
    inp.change(on_img, [inp, current, vals], [out])
    demo.load(choose, [inp, name], [current, *sliders, vals, formula, out])

if __name__ == "__main__":
    demo.launch()
