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

32 filters in `filters.py`, registered data-driven in `FILTERS` (name → function, params, and LaTeX formula shown in the UI). Self-check: `python filters.py`.

## Files

- `app.py` — Gradio UI (dropdown → dynamic sliders → live preview)
- `filters.py` — filter implementations
- `requirements.txt` — numpy, pillow, gradio
