"""Image filters from Gonzalez & Woods, Digital Image Processing (4th ed.).

All filters take a uint8 2D grayscale array and return a uint8 array.
FILTERS[name] = {"fn": fn(img, **params) -> uint8, "params": [param dicts],
                 "formula": LaTeX string rendered below the filter dropdown}
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

def negative(img, **kwargs):
    return (255 - np.asarray(img)).astype(np.uint8)


def log_transform(img, **kwargs):
    return _to_uint8(255 * np.log1p(np.asarray(img, dtype=float)) / np.log(256))


def power_law(img, gamma=0.4, **kwargs):
    return _to_uint8(255 * np.power(np.asarray(img, dtype=float) / 255, gamma))


def contrast_stretch(img, lo=0, hi=255, **kwargs):
    return _to_uint8((np.asarray(img, dtype=float) - lo) * 255 / max(hi - lo, 1))


def hist_eq(img, **kwargs):
    x = np.asarray(img)
    cdf = np.bincount(x.ravel(), minlength=256).cumsum()
    cdf = (cdf - cdf.min()) * 255 / max(cdf.max() - cdf.min(), 1)
    return cdf[x].astype(np.uint8)


def bit_plane(img, plane=7, **kwargs):
    return (((np.asarray(img) >> int(plane)) & 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ch3 - spatial smoothing
# ---------------------------------------------------------------------------

def box_filter(img, k=9, **kwargs):
    return _to_uint8(_convolve(img, np.ones((k, k)) / k ** 2))


def weighted_average(img, k=9, **kwargs):
    d = np.abs(np.arange(k) - (k - 1) / 2)
    w = (k + 1) / 2 - d
    w /= w.sum()
    return _to_uint8(_convolve(img, np.outer(w, w)))


def gaussian_blur(img, sigma=2.0, k=9, **kwargs):
    return _to_uint8(_convolve(img, _gauss_kernel(sigma, k)))


# ---------------------------------------------------------------------------
# Ch3 - spatial sharpening
# ---------------------------------------------------------------------------

def laplacian(img, scale=1.0, **kwargs):
    lap = _convolve(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    return _to_uint8(128 + scale * lap)  # centered view of the response


def _gradient_magnitude(img, gx_k, gy_k):
    gx = _convolve(img, gx_k)
    gy = _convolve(img, gy_k)
    return _norm_display(np.hypot(gx, gy))


def sobel(img, **kwargs):
    return _gradient_magnitude(
        img,
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    )


def prewitt(img, **kwargs):
    return _gradient_magnitude(
        img,
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    )


def _unsharp(img, sigma, amount, k):
    blur = _convolve(img, _gauss_kernel(sigma, k))
    return _as_float(img) + amount * (_as_float(img) - blur)


def _as_float(img):
    return np.asarray(img, dtype=float)


def unsharp_masking(img, sigma=2.0, amount=1.0, k=9, **kwargs):
    return _to_uint8(_unsharp(img, sigma, amount, k))


def high_boost(img, sigma=2.0, A=1.5, k=9, **kwargs):
    blur = _convolve(img, _gauss_kernel(sigma, k))
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


def _lp_mask(shape, D_0, kind, n):
    D = _freq_grid(shape)
    if kind == "ideal":
        return (D <= D_0).astype(float)
    if kind == "gaussian":
        return np.exp(-D ** 2 / (2 * D_0 ** 2))
    return 1 / (1 + (D / D_0) ** (2 * n))  # butterworth


def _apply_freq(img, kind, D_0, hp, n=2):
    H = _lp_mask(img.shape, D_0, kind, n)
    return _to_uint8(_ifft2(_fft2(img) * (1 - H if hp else H)))


def _freq_filter(kind, hp):
    def fn(img, D_0=0.1, n=2, **kwargs):
        return _apply_freq(img, kind, D_0, hp, n)
    return fn


def fft_magnitude(img, **kwargs):
    F = np.log1p(np.abs(_fft2(img)))
    return _norm_display(F)


def fft_phase(img, **kwargs):
    ang = np.angle(_fft2(img))  # -pi..pi
    return _to_uint8((ang + np.pi) * 255 / (2 * np.pi))


def fft_power(img, **kwargs):
    return _norm_display(np.log1p(np.abs(_fft2(img)) ** 2))


def _dct1(x):
    """Orthonormal DCT-II along the last axis (FFT-based, no scipy)."""
    N = x.shape[-1]
    y = np.concatenate([x, x[..., ::-1]], axis=-1)
    Y = np.fft.fft(y, axis=-1)[..., :N]
    C = np.real(Y * np.exp(-1j * np.pi * np.arange(N) / (2 * N)))
    C[..., 0] /= np.sqrt(2)
    C *= np.sqrt(1.0 / (2 * N))
    return C


def _dct2(img):
    return _dct1(_dct1(np.asarray(img, dtype=float)).T).T


def dct2_magnitude(img, **kwargs):
    return _norm_display(np.log1p(np.abs(_dct2(img))))


# ---------------------------------------------------------------------------
# Ch5 - noise + restoration
# ---------------------------------------------------------------------------

def add_gaussian_noise(img, sigma=20, **kwargs):
    return _to_uint8(np.asarray(img, dtype=float) + np.random.randn(*img.shape) * sigma)


def add_salt_pepper(img, p=0.05, **kwargs):
    out = np.asarray(img).copy()
    mask = np.random.random(img.shape) < p
    sp = np.random.random(img.shape) < 0.5
    out[mask & sp] = 0
    out[mask & ~sp] = 255
    return out


def arithmetic_mean(img, k=9, **kwargs):
    return np.mean(_windows(img, k), axis=0).astype(np.uint8)


def geometric_mean(img, k=9, **kwargs):
    w = np.where(_windows(img, k) == 0, 1, _windows(img, k))
    return np.exp(np.mean(np.log(w), axis=0)).astype(np.uint8)


def harmonic_mean(img, k=9, **kwargs):
    w = np.where(_windows(img, k) == 0, 1e-9, _windows(img, k))
    return (w.shape[0] / np.sum(1.0 / w, axis=0)).astype(np.uint8)


def contraharmonic_mean(img, k=9, Q=1.5, **kwargs):
    w = _windows(img, k)
    return (np.sum(w ** (Q + 1), axis=0) / np.sum(w ** Q, axis=0)).astype(np.uint8)


def order_statistic(img, kind, k=9, alpha=0.25, **kwargs):
    w = _windows(img, k)
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
        T = int(alpha * w.shape[0] / 2)
        out = np.mean(ws[T:ws.shape[0] - T], axis=0)
    return out.astype(np.uint8)


def _alpha_trimmed(img, k=9, alpha=0.25, **kwargs):
    return order_statistic(img, "alpha", k, alpha)


# ---------------------------------------------------------------------------
# error metrics between two images (a = reference, b = estimate)
# ---------------------------------------------------------------------------

def _pair(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def mse(a, b):
    a, b = _pair(a, b)
    return float(np.mean((a - b) ** 2))


def rmse(a, b):
    return float(np.sqrt(mse(a, b)))


def mae(a, b):
    a, b = _pair(a, b)
    return float(np.mean(np.abs(a - b)))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(255 ** 2 / m))


def snr(a, b):
    a, b = _pair(a, b)
    den = np.sum((a - b) ** 2)
    return float("inf") if den == 0 else float(10 * np.log10(np.sum(a ** 2) / den))


ERROR_METRICS = [
    ("MSE",  mse,  r"\text{MSE} = \frac{1}{MN} \sum (f - \hat f)^2"),
    ("RMSE", rmse, r"\text{RMSE} = \sqrt{\text{MSE}}"),
    ("MAE",  mae,  r"\text{MAE} = \frac{1}{MN} \sum |f - \hat f|"),
    ("PSNR", psnr, r"\text{PSNR} = 10 \log_{10} \frac{255^2}{\text{MSE}}"),
    ("SNR",  snr,  r"\text{SNR} = 10 \log_{10} \frac{\sum f^2}{\sum (f - \hat f)^2}"),
]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

KERNEL  = {"name": "k", "min": 3, "max": 15, "default": 9, "step": 2, "int": True}
GKERNEL = {"name": "k", "min": 3, "max": 31, "default": 9, "step": 2, "int": True}
SIGMA   = {"name": "sigma", "min": 0.5, "max": 10, "default": 2.0, "step": 0.1}
CUTOFF  = {"name": "D_0", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01}
ORDER   = {"name": "n", "min": 1, "max": 10, "default": 2, "step": 1, "int": True}

FILTERS = {
    # Ch3 - intensity
    "Ch3 · Negative":              {"fn": negative, "params": [], "formula": r"s = 255 - r"},
    "Ch3 · Log transform":         {"fn": log_transform, "params": [], "formula": r"s = c\,\log(1 + r),\qquad c = \frac{255}{\ln 256}"},
    "Ch3 · Power-law (gamma)":     {"fn": power_law, "params": [{"name": "gamma", "min": 0.1, "max": 5.0, "default": 0.4, "step": 0.1}], "formula": r"s = 255\left(\frac{r}{255}\right)^{\gamma}"},
    "Ch3 · Contrast stretch":      {"fn": contrast_stretch, "params": [
        {"name": "lo", "min": 0, "max": 127, "default": 0, "step": 1, "int": True},
        {"name": "hi", "min": 128, "max": 255, "default": 255, "step": 1, "int": True},
    ], "formula": r"s = 255\,\frac{r - lo}{hi - lo},\quad s \in [0, 255]"},
    "Ch3 · Histogram equalization": {"fn": hist_eq, "params": [], "formula": r"s_k = 255 \sum_{j=0}^{k} p_r(r_j),\qquad p_r(r_j) = \frac{n_j}{MN}"},
    "Ch3 · Bit-plane slicing":     {"fn": bit_plane, "params": [{"name": "plane", "min": 0, "max": 7, "default": 7, "step": 1, "int": True}], "formula": r"s = 255 \cdot \left( \left\lfloor \frac{r}{2^{plane}} \right\rfloor \bmod 2 \right)"},
    # Ch3 - smoothing
    "Ch3 · Box (mean)":            {"fn": box_filter, "params": [KERNEL], "formula": r"g(x,y) = \frac{1}{k^2} \sum_{(s,t) \in S_{xy}} f(s,t)"},
    "Ch3 · Weighted average":      {"fn": weighted_average, "params": [KERNEL], "formula": r"g(x,y) = \frac{\sum_{(s,t) \in S_{xy}} w(s,t)\, f(s,t)}{\sum w},\quad w(i) = \frac{k+1}{2} - \left| i - \frac{k-1}{2} \right|"},
    "Ch3 · Gaussian blur":         {"fn": gaussian_blur, "params": [SIGMA, GKERNEL], "formula": r"g = f \ast G,\qquad G(x,y) = \frac{1}{2\pi\sigma^2}\, e^{-\frac{x^2+y^2}{2\sigma^2}}"},
    # Ch3 - sharpening
    "Ch3 · Laplacian":             {"fn": laplacian, "params": [{"name": "scale", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}], "formula": r"\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2},\qquad g = 128 + scale \cdot \nabla^2 f"},
    "Ch3 · Sobel":                 {"fn": sobel, "params": [], "formula": r"G = \sqrt{G_x^2 + G_y^2},\quad G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} \ast f,\ G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} \ast f"},
    "Ch3 · Prewitt":               {"fn": prewitt, "params": [], "formula": r"G = \sqrt{G_x^2 + G_y^2},\quad G_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix} \ast f,\ G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix} \ast f"},
    "Ch3 · Unsharp masking":       {"fn": unsharp_masking, "params": [SIGMA, {"name": "amount", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}, GKERNEL], "formula": r"g = f + amount\,(f - \bar f)"},
    "Ch3 · High-boost":            {"fn": high_boost, "params": [SIGMA, {"name": "A", "min": 1.0, "max": 3.0, "default": 1.5, "step": 0.1}, GKERNEL], "formula": r"g = A\, f - \bar f"},
    # Ch4 - frequency
    "Ch4 · Ideal low-pass":        {"fn": _freq_filter("ideal", False), "params": [CUTOFF], "formula": r"H(u,v) = \begin{cases} 1 & D(u,v) \le D_0 \\ 0 & \text{otherwise} \end{cases}"},
    "Ch4 · Ideal high-pass":       {"fn": _freq_filter("ideal", True), "params": [CUTOFF], "formula": r"H(u,v) = \begin{cases} 1 & D(u,v) > D_0 \\ 0 & \text{otherwise} \end{cases}"},
    "Ch4 · Butterworth low-pass":  {"fn": _freq_filter("butterworth", False), "params": [CUTOFF, ORDER], "formula": r"H(u,v) = \frac{1}{1 + \left( D(u,v)/D_0 \right)^{2n}}"},
    "Ch4 · Butterworth high-pass": {"fn": _freq_filter("butterworth", True), "params": [CUTOFF, ORDER], "formula": r"H(u,v) = 1 - \frac{1}{1 + \left( D(u,v)/D_0 \right)^{2n}}"},
    "Ch4 · Gaussian low-pass":     {"fn": _freq_filter("gaussian", False), "params": [CUTOFF], "formula": r"H(u,v) = e^{-D^2(u,v)/(2D_0^2)}"},
    "Ch4 · Gaussian high-pass":    {"fn": _freq_filter("gaussian", True), "params": [CUTOFF], "formula": r"H(u,v) = 1 - e^{-D^2(u,v)/(2D_0^2)}"},
    "Ch4 · FFT magnitude":         {"fn": fft_magnitude, "params": [], "formula": r"\log\left( 1 + |F(u,v)| \right)"},
    "Ch4 · FFT phase":             {"fn": fft_phase, "params": [], "formula": r"\varphi(u,v) = \arctan\left( \frac{\operatorname{Im} F(u,v)}{\operatorname{Re} F(u,v)} \right)"},
    "Ch4 · FFT power spectrum":    {"fn": fft_power, "params": [], "formula": r"P(u,v) = |F(u,v)|^2,\qquad \log\left( 1 + P \right)"},
    "Ch4 · DCT (2D)":              {"fn": dct2_magnitude, "params": [], "formula": r"C(u,v) = \frac{2}{\sqrt{MN}}\, c_u c_v \sum_{x,y} f(x,y) \cos\frac{\pi u(2x+1)}{2M} \cos\frac{\pi v(2y+1)}{2N},\qquad \log(1+|C|)"},
    # Ch5 - noise + restoration
    "Ch5 · Add Gaussian noise":    {"fn": add_gaussian_noise, "params": [{"name": "sigma", "min": 1, "max": 100, "default": 20, "step": 1, "int": True}], "formula": r"g = f + \sigma\,\varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,1)"},
    "Ch5 · Add salt-and-pepper":   {"fn": add_salt_pepper, "params": [{"name": "p", "min": 0.01, "max": 0.5, "default": 0.05, "step": 0.01}], "formula": r"g = \begin{cases} 0 & \text{pepper (prob } p/2\text{)} \\ 255 & \text{salt (prob } p/2\text{)} \\ f & \text{otherwise} \end{cases}"},
    "Ch5 · Arithmetic mean":       {"fn": arithmetic_mean, "params": [KERNEL], "formula": r"\hat f = \frac{1}{k^2} \sum_{(s,t) \in S_{xy}} f(s,t)"},
    "Ch5 · Geometric mean":        {"fn": geometric_mean, "params": [KERNEL], "formula": r"\hat f = \left( \prod_{(s,t) \in S_{xy}} f(s,t) \right)^{1/k^2}"},
    "Ch5 · Harmonic mean":         {"fn": harmonic_mean, "params": [KERNEL], "formula": r"\hat f = \frac{k^2}{\sum_{(s,t) \in S_{xy}} \frac{1}{f(s,t)}}"},
    "Ch5 · Contraharmonic mean":   {"fn": contraharmonic_mean, "params": [KERNEL, {"name": "Q", "min": -1.5, "max": 1.5, "default": 1.5, "step": 0.1}], "formula": r"\hat f = \frac{\sum_{(s,t) \in S_{xy}} f(s,t)^{Q+1}}{\sum_{(s,t) \in S_{xy}} f(s,t)^{Q}}"},
    "Ch5 · Median":                {"fn": partial(order_statistic, kind="median"), "params": [KERNEL], "formula": r"\hat f = \operatorname*{median}_{(s,t) \in S_{xy}} \{ f(s,t) \}"},
    "Ch5 · Min":                   {"fn": partial(order_statistic, kind="min"), "params": [KERNEL], "formula": r"\hat f = \min_{(s,t) \in S_{xy}} \{ f(s,t) \}"},
    "Ch5 · Max":                   {"fn": partial(order_statistic, kind="max"), "params": [KERNEL], "formula": r"\hat f = \max_{(s,t) \in S_{xy}} \{ f(s,t) \}"},
    "Ch5 · Midpoint":              {"fn": partial(order_statistic, kind="midpoint"), "params": [KERNEL], "formula": r"\hat f = \tfrac{1}{2}\left( \min_{S_{xy}}\{ f \} + \max_{S_{xy}}\{ f \} \right)"},
    "Ch5 · Alpha-trimmed":         {"fn": _alpha_trimmed, "params": [KERNEL, {"name": "alpha", "min": 0.05, "max": 0.45, "default": 0.25, "step": 0.05}], "formula": r"\hat f = \frac{1}{k^2 - 2T} \sum_{i=T+1}^{k^2-T} f_{(i)},\qquad T = \left\lfloor \frac{\alpha k^2}{2} \right\rfloor"},
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

    # other frequency views + DCT (reference orthonormal DCT-II for the check)
    assert np.unique(fft_phase(np.full((8, 8), 100, np.uint8))).size == 1  # constant -> zero phase
    assert fft_power(x).max() == 255
    a = np.random.default_rng(0).random((5, 6))
    def _dct_ref(img):
        M, N = img.shape
        def D(n):
            u = np.arange(n)[:, None]; x = np.arange(n)[None, :]
            c = np.where(u == 0, 1 / np.sqrt(2), 1.0)
            return np.sqrt(2 / n) * c * np.cos(np.pi * u * (2 * x + 1) / (2 * n))
        return D(M) @ img @ D(N).T
    assert np.abs(_dct2(a) - _dct_ref(a)).max() < 1e-9

    # error metrics
    a = np.arange(16, dtype=float).reshape(4, 4); b = a + 10
    assert mse(a, b) == 100.0
    assert rmse(a, b) == 10.0
    assert mae(a, b) == 10.0
    assert psnr(a, b) == 10 * np.log10(255 ** 2 / 100)
    assert snr(a, b) == 10 * np.log10(np.sum(a ** 2) / (16 * 100))
    assert psnr(a, a) == float("inf")

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

    assert all(f.get("formula") for f in FILTERS.values())
    print(f"filters OK ({len(FILTERS)} filters)")


if __name__ == "__main__":
    demo()
