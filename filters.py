"""Image filters from Gonzalez & Woods, Digital Image Processing (4th ed.).

All filters take a uint8 2D grayscale array and return a uint8 array.
FILTERS[name] = {"fn": fn(img, **params) -> uint8, "params": [param dicts]}
param dict: {"name", "min", "max", "default", "step", "int"?}
"""

import numpy as np
from functools import partial

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _windows(img, k):
    """Stack all k x k neighborhoods as (k*k, H, W) float32 array, edge-padded."""
    k = int(k)
    p = np.pad(np.asarray(img, dtype=np.float32), k // 2, mode="edge")
    h, w = img.shape
    stack = np.empty((k * k, h, w), dtype=np.float32)
    i = 0
    for dy in range(k):
        for dx in range(k):
            stack[i] = p[dy:dy + h, dx:dx + w]
            i += 1
    return stack


def _convolve(img, kernel):
    """2D correlation with odd-sized kernel, edge padding (numpy, no scipy)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(np.asarray(img, dtype=np.float64), ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros(img.shape)
    for dy in range(kh):
        for dx in range(kw):
            out += kernel[dy, dx] * p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out


def _gauss_kernel(sigma, k):
    ax = np.arange(-(k // 2), k // 2 + 1, dtype=float)
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def _to_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _norm_display(mag):
    mx = mag.max()
    return _to_uint8(mag * 255 / mx) if mx > 0 else np.zeros_like(mag, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Ch3 - intensity transformations
# ---------------------------------------------------------------------------

def negative(img, **p):
    return (255 - np.asarray(img)).astype(np.uint8)


def log_transform(img, **p):
    return _to_uint8(255 * np.log1p(np.asarray(img, dtype=float)) / np.log(256))


def power_law(img, gamma=0.4, **p):
    return _to_uint8(255 * np.power(np.asarray(img, dtype=float) / 255, gamma))


def contrast_stretch(img, lo=0, hi=255, **p):
    return _to_uint8((np.asarray(img, dtype=float) - lo) * 255 / max(hi - lo, 1))


def hist_eq(img, **p):
    x = np.asarray(img)
    cdf = np.bincount(x.ravel(), minlength=256).cumsum()
    cdf = (cdf - cdf.min()) * 255 / max(cdf.max() - cdf.min(), 1)
    return cdf[x].astype(np.uint8)


def bit_plane(img, plane=7, **p):
    return (((np.asarray(img) >> int(plane)) & 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ch3 - spatial smoothing
# ---------------------------------------------------------------------------

def box_filter(img, kernel=9, **p):
    return _to_uint8(_convolve(img, np.ones((kernel, kernel)) / kernel ** 2))


def weighted_average(img, kernel=9, **p):
    d = np.abs(np.arange(kernel) - (kernel - 1) / 2)
    w = (kernel + 1) / 2 - d
    w /= w.sum()
    return _to_uint8(_convolve(img, np.outer(w, w)))


def gaussian_blur(img, sigma=2.0, kernel=9, **p):
    return _to_uint8(_convolve(img, _gauss_kernel(sigma, kernel)))


# ---------------------------------------------------------------------------
# Ch3 - spatial sharpening
# ---------------------------------------------------------------------------

def laplacian(img, scale=1.0, **p):
    lap = _convolve(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    return _to_uint8(128 + scale * lap)  # centered view of the response


def _gradient_magnitude(img, gx_k, gy_k):
    gx = _convolve(img, gx_k)
    gy = _convolve(img, gy_k)
    return _norm_display(np.hypot(gx, gy))


def sobel(img, **p):
    return _gradient_magnitude(
        img,
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    )


def prewitt(img, **p):
    return _gradient_magnitude(
        img,
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    )


def _unsharp(img, sigma, amount, kernel):
    blur = _convolve(img, _gauss_kernel(sigma, kernel))
    return _as_float(img) + amount * (_as_float(img) - blur)


def _as_float(img):
    return np.asarray(img, dtype=float)


def unsharp_masking(img, sigma=2.0, amount=1.0, kernel=9, **p):
    return _to_uint8(_unsharp(img, sigma, amount, kernel))


def high_boost(img, sigma=2.0, A=1.5, kernel=9, **p):
    blur = _convolve(img, _gauss_kernel(sigma, kernel))
    return _to_uint8(A * _as_float(img) - blur)


# ---------------------------------------------------------------------------
# Ch4 - frequency domain
# ---------------------------------------------------------------------------

def _fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))


def _ifft2(F):
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def _freq_grid(shape):
    m, n = shape
    u = np.fft.fftshift(np.fft.fftfreq(m))[:, None]
    v = np.fft.fftshift(np.fft.fftfreq(n))[None, :]
    return np.sqrt(u ** 2 + v ** 2)


def _lp_mask(shape, cutoff, kind, order):
    D = _freq_grid(shape)
    if kind == "ideal":
        return (D <= cutoff).astype(float)
    if kind == "gaussian":
        return np.exp(-D ** 2 / (2 * cutoff ** 2))
    return 1 / (1 + (D / cutoff) ** (2 * order))  # butterworth


def _apply_freq(img, kind, cutoff, hp, order=2):
    H = _lp_mask(img.shape, cutoff, kind, order)
    return _to_uint8(_ifft2(_fft2(img) * (1 - H if hp else H)))


def _freq_filter(kind, hp):
    def fn(img, cutoff=0.1, order=2, **p):
        return _apply_freq(img, kind, cutoff, hp, order)
    return fn


def fft_magnitude(img, **p):
    F = np.log1p(np.abs(_fft2(img)))
    return _norm_display(F)


# ---------------------------------------------------------------------------
# Ch5 - noise + restoration
# ---------------------------------------------------------------------------

def add_gaussian_noise(img, sigma=20, **p):
    return _to_uint8(np.asarray(img, dtype=float) + np.random.randn(*img.shape) * sigma)


def add_salt_pepper(img, prob=0.05, **p):
    out = np.asarray(img).copy()
    mask = np.random.random(img.shape) < prob
    sp = np.random.random(img.shape) < 0.5
    out[mask & sp] = 0
    out[mask & ~sp] = 255
    return out


def arithmetic_mean(img, kernel=9, **p):
    return np.mean(_windows(img, kernel), axis=0).astype(np.uint8)


def geometric_mean(img, kernel=9, **p):
    w = np.where(_windows(img, kernel) == 0, 1, _windows(img, kernel))
    return np.exp(np.mean(np.log(w), axis=0)).astype(np.uint8)


def harmonic_mean(img, kernel=9, **p):
    w = np.where(_windows(img, kernel) == 0, 1e-9, _windows(img, kernel))
    return (w.shape[0] / np.sum(1.0 / w, axis=0)).astype(np.uint8)


def contraharmonic_mean(img, kernel=9, Q=1.5, **p):
    w = _windows(img, kernel)
    return (np.sum(w ** (Q + 1), axis=0) / np.sum(w ** Q, axis=0)).astype(np.uint8)


def order_statistic(img, kind, kernel=9, alpha=0.25, **p):
    w = _windows(img, kernel)
    if kind == "median":
        out = np.median(w, axis=0)
    elif kind == "min":
        out = w.min(axis=0)
    elif kind == "max":
        out = w.max(axis=0)
    elif kind == "midpoint":
        out = (w.min(axis=0) + w.max(axis=0)) / 2
    else:  # alpha-trimmed
        ws = np.sort(w, axis=0)
        t = int(alpha * w.shape[0] / 2)
        out = np.mean(ws[t:ws.shape[0] - t], axis=0)
    return out.astype(np.uint8)


def _alpha_trimmed(img, kernel=9, alpha=0.25, **p):
    return order_statistic(img, "alpha", kernel, alpha)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

KERNEL  = {"name": "kernel size", "min": 3, "max": 15, "default": 9, "step": 2, "int": True}
GKERNEL = {"name": "kernel size", "min": 3, "max": 31, "default": 9, "step": 2, "int": True}
SIGMA   = {"name": "sigma", "min": 0.5, "max": 10, "default": 2.0, "step": 0.1}
CUTOFF  = {"name": "cutoff", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01}
ORDER   = {"name": "order", "min": 1, "max": 10, "default": 2, "step": 1, "int": True}

FILTERS = {
    # Ch3 - intensity
    "Ch3 · Negative":              {"fn": negative, "params": []},
    "Ch3 · Log transform":         {"fn": log_transform, "params": []},
    "Ch3 · Power-law (gamma)":     {"fn": power_law, "params": [{"name": "gamma", "min": 0.1, "max": 5.0, "default": 0.4, "step": 0.1}]},
    "Ch3 · Contrast stretch":      {"fn": contrast_stretch, "params": [
        {"name": "lo", "min": 0, "max": 127, "default": 0, "step": 1, "int": True},
        {"name": "hi", "min": 128, "max": 255, "default": 255, "step": 1, "int": True},
    ]},
    "Ch3 · Histogram equalization": {"fn": hist_eq, "params": []},
    "Ch3 · Bit-plane slicing":     {"fn": bit_plane, "params": [{"name": "plane", "min": 0, "max": 7, "default": 7, "step": 1, "int": True}]},
    # Ch3 - smoothing
    "Ch3 · Box (mean)":            {"fn": box_filter, "params": [KERNEL]},
    "Ch3 · Weighted average":      {"fn": weighted_average, "params": [KERNEL]},
    "Ch3 · Gaussian blur":         {"fn": gaussian_blur, "params": [SIGMA, GKERNEL]},
    # Ch3 - sharpening
    "Ch3 · Laplacian":             {"fn": laplacian, "params": [{"name": "scale", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}]},
    "Ch3 · Sobel":                 {"fn": sobel, "params": []},
    "Ch3 · Prewitt":               {"fn": prewitt, "params": []},
    "Ch3 · Unsharp masking":       {"fn": unsharp_masking, "params": [SIGMA, {"name": "amount", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}, GKERNEL]},
    "Ch3 · High-boost":            {"fn": high_boost, "params": [SIGMA, {"name": "A", "min": 1.0, "max": 3.0, "default": 1.5, "step": 0.1}, GKERNEL]},
    # Ch4 - frequency
    "Ch4 · Ideal low-pass":        {"fn": _freq_filter("ideal", False), "params": [CUTOFF]},
    "Ch4 · Ideal high-pass":       {"fn": _freq_filter("ideal", True), "params": [CUTOFF]},
    "Ch4 · Butterworth low-pass":  {"fn": _freq_filter("butterworth", False), "params": [CUTOFF, ORDER]},
    "Ch4 · Butterworth high-pass": {"fn": _freq_filter("butterworth", True), "params": [CUTOFF, ORDER]},
    "Ch4 · Gaussian low-pass":     {"fn": _freq_filter("gaussian", False), "params": [CUTOFF]},
    "Ch4 · Gaussian high-pass":    {"fn": _freq_filter("gaussian", True), "params": [CUTOFF]},
    "Ch4 · FFT magnitude":         {"fn": fft_magnitude, "params": []},
    # Ch5 - noise + restoration
    "Ch5 · Add Gaussian noise":    {"fn": add_gaussian_noise, "params": [{"name": "sigma", "min": 1, "max": 100, "default": 20, "step": 1, "int": True}]},
    "Ch5 · Add salt-and-pepper":   {"fn": add_salt_pepper, "params": [{"name": "prob", "min": 0.01, "max": 0.5, "default": 0.05, "step": 0.01}]},
    "Ch5 · Arithmetic mean":       {"fn": arithmetic_mean, "params": [KERNEL]},
    "Ch5 · Geometric mean":        {"fn": geometric_mean, "params": [KERNEL]},
    "Ch5 · Harmonic mean":         {"fn": harmonic_mean, "params": [KERNEL]},
    "Ch5 · Contraharmonic mean":   {"fn": contraharmonic_mean, "params": [KERNEL, {"name": "Q", "min": -1.5, "max": 1.5, "default": 1.5, "step": 0.1}]},
    "Ch5 · Median":                {"fn": partial(order_statistic, kind="median"), "params": [KERNEL]},
    "Ch5 · Min":                   {"fn": partial(order_statistic, kind="min"), "params": [KERNEL]},
    "Ch5 · Max":                   {"fn": partial(order_statistic, kind="max"), "params": [KERNEL]},
    "Ch5 · Midpoint":              {"fn": partial(order_statistic, kind="midpoint"), "params": [KERNEL]},
    "Ch5 · Alpha-trimmed":         {"fn": _alpha_trimmed, "params": [KERNEL, {"name": "alpha", "min": 0.05, "max": 0.45, "default": 0.25, "step": 0.05}]},
}


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def demo():
    x = np.arange(9, dtype=np.uint8).reshape(3, 3)  # [[0..8]]
    assert negative(x)[0, 0] == 255
    assert log_transform(x)[0, 0] == 0
    assert power_law(x, 1.0).tolist() == x.tolist()
    assert contrast_stretch(x, 2, 6)[1, 1] == 127  # (4-2)*255/4 = 127.5 -> 127
    c = np.full((4, 4), 50, dtype=np.uint8)
    assert np.unique(hist_eq(c)).size == 1
    assert bit_plane(x, 0)[0, 1] == 255  # pixel value 1 has bit0 set

    assert box_filter(x, 3)[1, 1] == 4
    assert weighted_average(x, 3)[1, 1] == 4
    assert abs(_gauss_kernel(2, 9).sum() - 1) < 1e-9

    assert laplacian(np.full((3, 3), 100, np.uint8))[0, 0] == 128  # zero response -> mid-gray
    assert sobel(np.full((5, 5), 100, np.uint8)).max() == 0
    assert unsharp_masking(x, 1.0, 0.0)[1, 1] == 4  # amount 0 -> original

    # FFT roundtrip + near-identity low-pass at max cutoff
    assert np.allclose(_ifft2(_fft2(x.astype(float))), x.astype(float), atol=1e-6)
    assert np.abs(_apply_freq(x, "gaussian", 0.5, False).astype(int) - x).max() <= 2

    np.random.seed(0)
    y = add_salt_pepper(np.full((100, 100), 128, np.uint8), 0.1)
    frac = np.mean((y == 0) | (y == 255))
    assert 0.07 < frac < 0.13

    assert arithmetic_mean(x, 3)[1, 1] == 4
    assert harmonic_mean(x + 1, 3)[1, 1] == 3  # harm. mean of 1..9 ≈ 3.18, truncated
    assert contraharmonic_mean(x, 3, 0.0)[1, 1] == 4  # Q=0 -> arithmetic mean
    assert order_statistic(x, "median", 3)[1, 1] == 4
    assert order_statistic(x, "min", 3)[1, 1] == 0
    assert order_statistic(x, "max", 3)[1, 1] == 8
    assert order_statistic(x, "midpoint", 3)[1, 1] == 4
    assert np.abs(_alpha_trimmed(x, 3, 0.25)[1, 1] - 4) <= 1

    print(f"filters OK ({len(FILTERS)} filters)")


if __name__ == "__main__":
    demo()
