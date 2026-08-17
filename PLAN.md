# Plan: Image Filter Demo

Apply image-processing filters from *Digital Image Processing* (Gonzalez & Woods, 4th ed.)
to an image. User picks a filter from a dropdown, tunes its parameters with sliders,
sees the result live.

## Stack

| Concern | Choice | Why |
|---|---|---|
| UI | Gradio | tkinter not installed; Gradio gives dropdown + slider + image natively, in-browser |
| Math | numpy | required — pure-Python loops too slow for live slider previews; `numpy.fft` covers frequency domain |
| I/O | Pillow | already installed |

- `pip install numpy gradio`
- No cv2/skimage — they ship the filters pre-built, defeating the point of implementing the book.
- No scipy — `numpy.fft` is enough.

## Files

```
ImageProcessingFilterDemo/
├── app.py            # Gradio UI: dropdown → dynamic sliders → preview
├── filters.py        # filter implementations (numpy), registry + demo() self-checks
├── requirements.txt  # numpy, pillow, gradio
└── image/            # sample image(s)
```

## Core design: data-driven registry

`filters.py` holds one dict. Each entry = `fn` + `params`. `app.py` is thin: it renders
sliders from the registry and calls `fn(gray, **params)` on change. Adding a filter is one
dict entry — no UI code changes.

```python
FILTERS = {
  "Ch3 · Gaussian blur": {
      "fn": gaussian_blur,
      "params": [
          {"name": "sigma",  "min": 0.5, "max": 10, "default": 2.0, "step": 0.1},
          {"name": "kernel", "min": 3,   "max": 51, "default": 9,   "step": 2},
      ],
  },
}
```

## Filter catalog

**v1 (first cut):**

- Ch 3 — Intensity: negative, log, power-law (gamma), contrast stretching,
  histogram equalization, bit-plane slicing
- Ch 3 — Spatial smoothing: box (mean), weighted average, median, Gaussian
- Ch 3 — Spatial sharpening: Laplacian, Sobel, Prewitt, unsharp masking, high-boost
- Ch 4 — Frequency: ideal / Butterworth / Gaussian LPF + HPF (cutoff slider), FFT magnitude view
- Ch 5 — Noise + restoration: add Gaussian / salt-and-pepper noise; arithmetic, geometric,
  harmonic, contraharmonic mean; median, min, max, midpoint, alpha-trimmed

**Later (same registry, more entries):**

- histogram matching, bilateral, homomorphic, Wiener, grayscale morphology
  (erode / dilate / open / close)

## Build order

1. `filters.py` — v1 filters, each with a `demo()` self-check (numpy `assert` on a tiny array)
2. `app.py` — dropdown + dynamic sliders + preview wired to the registry
3. `requirements.txt` + run instructions
