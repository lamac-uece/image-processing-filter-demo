# Image Processing Filter Demo

Interactive demo of image filters from Gonzalez & Woods, *Digital Image Processing* (4th ed.) — intensity/contrast transforms, spatial and frequency-domain filters, and noise/restoration.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # first time only
pip install -r requirements.txt
python app.py
```

Opens a Gradio UI: pick a filter from the dropdown, adjust the sliders, see a live preview. `image/Lenna.png` is a ready sample input.

## Filters

42 filters in `filters.py` (Ch2 resolution/depth, Ch3 intensity/spatial, Ch4 frequency, Ch5 noise/restoration), registered data-driven in `FILTERS` (name → function, params, and LaTeX formula shown in the UI). Self-check: `python filters.py`.

## Classroom features

- Chapter tabs (Ch2–Ch5) with a filter dropdown per chapter
- One-line plain-language caption per filter (PT)
- Parameter presets ("Try a preset") for filters with classic settings
- Reset-to-defaults button
- Input/result gray-level histograms and a spatial difference map
- Error PDF, 10 quality metrics, and entropy in the metrics panel
- Deep links: `?filter=<name>&p0=<v>&p1=<v>` restores state on load (shown as a copyable link)

## Files

- `app.py` — Gradio UI (dropdown → dynamic sliders → live preview)
- `filters.py` — filter implementations
- `requirements.txt` — numpy, pillow, gradio
