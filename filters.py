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
# Ch2 - resolution & bit depth (handout exercises 1-2)
# ---------------------------------------------------------------------------

def downsample(img, f=4, **kwargs):
    """Nearest-neighbor downsample by f, then upsample back to the input size.

    Pads the input to a multiple of f first, so the result always has the same
    shape as the input (handout: images the same size as the original).
    """
    x = np.asarray(img)
    f = int(f)
    h, w = x.shape
    xp = np.pad(x, ((0, (-h) % f), (0, (-w) % f)), mode="edge")
    return np.kron(xp[::f, ::f], np.ones((f, f), dtype=np.uint8))[:h, :w]


def bit_depth_reduce(img, b=4, **kwargs):
    """Quantize gray levels to 2**b (full-range scaling, b in [1, 7])."""
    b = int(b)
    x = np.asarray(img, dtype=np.float64)
    q = np.round(x * (2 ** b - 1) / 255) * (255.0 / (2 ** b - 1))
    return q.astype(np.uint8)


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

def add_gaussian_noise(img, sigma=20, seed=0, **kwargs):
    rng = np.random.RandomState(int(seed))
    return _to_uint8(np.asarray(img, dtype=float) + rng.randn(*img.shape) * sigma)


def add_salt_pepper(img, p=0.05, seed=0, **kwargs):
    rng = np.random.RandomState(int(seed))
    out = np.asarray(img).copy()
    mask = rng.random(img.shape) < p
    sp = rng.random(img.shape) < 0.5
    out[mask & sp] = 0
    out[mask & ~sp] = 255
    return out


def add_uniform_noise(img, A=30, seed=0, **kwargs):
    """Zero-mean uniform noise in [-A, A] (handout: p(z) = 1/(2A))."""
    rng = np.random.RandomState(int(seed))
    return _to_uint8(np.asarray(img, dtype=float) + rng.uniform(-A, A, img.shape))


def add_erlang_noise(img, a=0.1, b=2, seed=0, **kwargs):
    """Zero-mean Erlang/Gamma noise: shape b, rate a (handout: mean b/a)."""
    rng = np.random.RandomState(int(seed))
    z = rng.gamma(b, 1.0 / a, img.shape) - b / a
    return _to_uint8(np.asarray(img, dtype=float) + z)


def add_exponential_noise(img, a=0.1, seed=0, **kwargs):
    """Zero-mean exponential noise, rate a (Erlang with b=1; mean 1/a)."""
    rng = np.random.RandomState(int(seed))
    z = rng.exponential(1.0 / a, img.shape) - 1.0 / a
    return _to_uint8(np.asarray(img, dtype=float) + z)


def add_rayleigh_noise(img, b=200, seed=0, **kwargs):
    """Zero-mean Rayleigh noise (handout: scale sigma = sqrt(b/2), mean sqrt(pi*b)/2)."""
    rng = np.random.RandomState(int(seed))
    z = rng.rayleigh(np.sqrt(b / 2.0), img.shape) - 0.5 * np.sqrt(np.pi * b)
    return _to_uint8(np.asarray(img, dtype=float) + z)


def add_poisson_noise(img, mu=50, seed=0, **kwargs):
    """Zero-mean Poisson noise, intensity mu (mean = variance = mu)."""
    rng = np.random.RandomState(int(seed))
    z = rng.poisson(mu, img.shape).astype(np.float64) - mu
    return _to_uint8(np.asarray(img, dtype=float) + z)


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
# information measures
# ---------------------------------------------------------------------------

def entropy(img):
    """Shannon entropy of the gray-level histogram, in bits."""
    p = np.bincount(np.asarray(img).ravel(), minlength=256)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


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


def me(a, b):
    """Maximum error (handout)."""
    a, b = _pair(a, b)
    return float(np.max(np.abs(a - b)))


def nmse(a, b):
    """Normalized mean square error (handout)."""
    a, b = _pair(a, b)
    den = np.sum(a ** 2)
    return float("inf") if den == 0 else float(np.sum((a - b) ** 2) / den)


def covariance(a, b):
    """Covariance sigma_fg (handout)."""
    a, b = _pair(a, b)
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def correlation(a, b):
    """Correlation coefficient rho = sigma_fg / (sigma_f * sigma_g) (handout)."""
    a, b = _pair(a, b)
    num = np.sum((a - a.mean()) * (b - b.mean()))
    den = np.sqrt(np.sum((a - a.mean()) ** 2) * np.sum((b - b.mean()) ** 2))
    if den == 0:
        return 1.0 if np.all(a == b) else 0.0
    return float(num / den)


def jaccard(a, b, delta=1):
    """Fraction of pixels equal within tolerance delta (handout)."""
    a, b = _pair(a, b)
    return float(np.mean(np.abs(a - b) <= delta))


ERROR_METRICS = [
    ("MSE",     mse,        r"\text{MSE} = \frac{1}{MN} \sum (f - \hat f)^2"),
    ("RMSE",    rmse,       r"\text{RMSE} = \sqrt{\text{MSE}}"),
    ("MAE",     mae,        r"\text{MAE} = \frac{1}{MN} \sum |f - \hat f|"),
    ("ME",      me,         r"\text{ME} = \max |f - \hat f|"),
    ("NMSE",    nmse,       r"\text{NMSE} = \frac{\sum (f-\hat f)^2}{\sum f^2}"),
    ("PSNR",    psnr,       r"\text{PSNR} = 10 \log_{10} \frac{255^2}{\text{MSE}}"),
    ("SNR",     snr,        r"\text{SNR} = 10 \log_{10} \frac{\sum f^2}{\sum (f - \hat f)^2}"),
    ("Cov",     covariance, r"\sigma_{f\hat f} = \frac{1}{MN}\sum (f-\mu_f)(\hat f-\mu_{\hat f})"),
    ("Corr",    correlation, r"\rho = \frac{\sigma_{f\hat f}}{\sigma_f \sigma_{\hat f}}"),
    ("Jaccard", jaccard,    r"J = \frac{1}{MN}\sum \mathbf{1}[\,|f-\hat f| \le 1\,]"),
]


# ---------------------------------------------------------------------------
# classroom metadata: one-line captions and parameter presets per filter
# ---------------------------------------------------------------------------

CAPTIONS = {
    "Ch2 · Downsample (resolution)": "Reduces spatial resolution (f×f blocks) — the larger f, the more pixelated.",
    "Ch2 · Bit-depth reduction": "Lowers gray levels from 2⁸ to 2ᵇ — banding becomes visible at low b.",
    "Ch3 · Negative": "Inverts intensities: black becomes white and vice versa.",
    "Ch3 · Log transform": "Brightens shadows while compressing highlights — useful for wide dynamic range.",
    "Ch3 · Power-law (gamma)": "γ<1 brightens shadows, γ>1 darkens — the gamma adjustment of monitors.",
    "Ch3 · Contrast stretch": "Expands the interval [lo, hi] to [0, 255] — enhances contrast of the chosen range.",
    "Ch3 · Histogram equalization": "Respreads the gray levels to use the whole range — equalizes the histogram.",
    "Ch3 · Bit-plane slicing": "Isolates one of the 8 bits of each pixel — high planes give the shape, low ones fine detail.",
    "Ch3 · Box (mean)": "Arithmetic mean in a k×k window — smooths/blurs (low-pass).",
    "Ch3 · Weighted average": "Weighted mean (heavier center) — smooths while keeping more detail.",
    "Ch3 · Gaussian blur": "Convolution with a Gaussian kernel — soft blur, the basis of noise reduction.",
    "Ch3 · Laplacian": "Second derivative — highlights edges; scale controls the intensity.",
    "Ch3 · Sobel": "Gradient magnitude — emphasizes contours in both directions.",
    "Ch3 · Prewitt": "Like Sobel, with uniform weights — simpler, slightly less robust.",
    "Ch3 · Unsharp masking": "Adds the detail (f − blurred) back to the image — sharpening; amount controls the strength.",
    "Ch3 · High-boost": "f = A·original − blurred — A>1 emphasizes edges, A<1 smooths.",
    "Ch4 · Ideal low-pass": "Cuts frequencies above D₀ all at once — blurs and can create rings (ringing).",
    "Ch4 · Ideal high-pass": "Keeps only high frequencies — highlights edges and removes the background.",
    "Ch4 · Butterworth low-pass": "Smooth transition controlled by n — less ringing than the ideal.",
    "Ch4 · Butterworth high-pass": "High-pass with a smooth transition — edges without the ideal filter's ring.",
    "Ch4 · Gaussian low-pass": "Gaussian attenuation — soft blur with no ringing.",
    "Ch4 · Gaussian high-pass": "Gaussian high-pass — edges and fine textures, dark background.",
    "Ch4 · FFT magnitude": "Fourier spectrum (log scale): the center is the low frequency (overall brightness).",
    "Ch4 · FFT phase": "Spectrum phase — carries the structure/position of the edges.",
    "Ch4 · FFT power spectrum": "|F|² on a log scale — emphasis on the dominant frequencies.",
    "Ch4 · DCT (2D)": "2D cosine transform (used in JPEG) — energy concentrated in the first coefficients.",
    "Ch5 · Add Gaussian noise": "Adds Gaussian noise N(0, σ) — the classic model of electronic noise.",
    "Ch5 · Add salt-and-pepper": "Random pixels become 0 or 255 — the impulsive noise of old photos.",
    "Ch5 · Add uniform noise": "Uniform noise in [−A, A] — every value with the same probability.",
    "Ch5 · Add Erlang (gamma) noise": "Noise with an Erlang/gamma distribution (a = rate, b = shape) — asymmetric.",
    "Ch5 · Add exponential noise": "Exponential noise — a special case of Erlang with b = 1.",
    "Ch5 · Add Rayleigh noise": "Rayleigh noise (parameter b) — typical of radar/ultrasound.",
    "Ch5 · Add Poisson noise": "Poisson noise (intensity µ) — inherent to photon counting in low light.",
    "Ch5 · Arithmetic mean": "Window mean — removes Gaussian noise but blurs the edges.",
    "Ch5 · Geometric mean": "Geometric mean — removes Gaussian noise while keeping a bit more detail.",
    "Ch5 · Harmonic mean": "Harmonic mean — great against pepper, poor against salt.",
    "Ch5 · Contraharmonic mean": "Q>0 removes pepper, Q<0 removes salt — generalizes the mean.",
    "Ch5 · Median": "Window median — the champion against salt-and-pepper, preserves edges.",
    "Ch5 · Min": "Window minimum — removes salt (white dots).",
    "Ch5 · Max": "Window maximum — removes pepper (black dots).",
    "Ch5 · Midpoint": "Mean of min and max — useful for uniform/Gaussian noise.",
    "Ch5 · Alpha-trimmed": "Discards the α/2 smallest and largest before averaging — a middle ground between mean and median.",
}

PRESETS = {
    "Ch2 · Downsample (resolution)": [
        {"label": "256×256 (f=2)", "values": {"f": 2}},
        {"label": "128×128 (f=4)", "values": {"f": 4}},
        {"label": "32×32 (f=16)", "values": {"f": 16}},
    ],
    "Ch2 · Bit-depth reduction": [
        {"label": "b=1 (black/white)", "values": {"b": 1}},
        {"label": "b=4 (16 levels)", "values": {"b": 4}},
    ],
    "Ch3 · Power-law (gamma)": [
        {"label": "γ=0.4 (brighten)", "values": {"gamma": 0.4}},
        {"label": "γ=2.5 (darken)", "values": {"gamma": 2.5}},
    ],
    "Ch3 · Contrast stretch": [
        {"label": "identity (0–255)", "values": {"lo": 0, "hi": 255}},
        {"label": "focus middle (64–192)", "values": {"lo": 64, "hi": 192}},
    ],
    "Ch3 · Bit-plane slicing": [
        {"label": "plane 7 (MSB)", "values": {"plane": 7}},
        {"label": "plane 0 (LSB)", "values": {"plane": 0}},
    ],
    "Ch3 · Box (mean)": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch3 · Weighted average": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch3 · Gaussian blur": [
        {"label": "soft (σ=1, k=7)", "values": {"sigma": 1.0, "k": 7}},
        {"label": "strong (σ=5, k=21)", "values": {"sigma": 5.0, "k": 21}},
    ],
    "Ch3 · Laplacian": [{"label": "scale=1", "values": {"scale": 1.0}}, {"label": "scale=3", "values": {"scale": 3.0}}],
    "Ch3 · Unsharp masking": [
        {"label": "light (amount=1)", "values": {"sigma": 2.0, "amount": 1.0, "k": 9}},
        {"label": "strong (amount=3)", "values": {"sigma": 2.0, "amount": 3.0, "k": 15}},
    ],
    "Ch3 · High-boost": [
        {"label": "A=1.5", "values": {"sigma": 2.0, "A": 1.5, "k": 9}},
        {"label": "A=2.5", "values": {"sigma": 2.0, "A": 2.5, "k": 15}},
    ],
    "Ch4 · Ideal low-pass": [{"label": "narrow (D₀=0.05)", "values": {"D_0": 0.05}}, {"label": "wide (D₀=0.3)", "values": {"D_0": 0.3}}],
    "Ch4 · Ideal high-pass": [{"label": "narrow (D₀=0.05)", "values": {"D_0": 0.05}}, {"label": "wide (D₀=0.3)", "values": {"D_0": 0.3}}],
    "Ch4 · Butterworth low-pass": [{"label": "D₀=0.05", "values": {"D_0": 0.05}}, {"label": "D₀=0.3", "values": {"D_0": 0.3}}],
    "Ch4 · Butterworth high-pass": [{"label": "D₀=0.05", "values": {"D_0": 0.05}}, {"label": "D₀=0.3", "values": {"D_0": 0.3}}],
    "Ch4 · Gaussian low-pass": [{"label": "D₀=0.05", "values": {"D_0": 0.05}}, {"label": "D₀=0.3", "values": {"D_0": 0.3}}],
    "Ch4 · Gaussian high-pass": [{"label": "D₀=0.05", "values": {"D_0": 0.05}}, {"label": "D₀=0.3", "values": {"D_0": 0.3}}],
    "Ch5 · Add Gaussian noise": [{"label": "σ=10", "values": {"sigma": 10}}, {"label": "σ=50", "values": {"sigma": 50}}],
    "Ch5 · Add salt-and-pepper": [{"label": "p=0.05", "values": {"p": 0.05}}, {"label": "p=0.3", "values": {"p": 0.3}}],
    "Ch5 · Add uniform noise": [{"label": "A=10", "values": {"A": 10}}, {"label": "A=60", "values": {"A": 60}}],
    "Ch5 · Add Erlang (gamma) noise": [
        {"label": "a=0.2, b=2", "values": {"a": 0.2, "b": 2}},
        {"label": "a=0.05, b=5", "values": {"a": 0.05, "b": 5}},
    ],
    "Ch5 · Add exponential noise": [{"label": "a=0.2", "values": {"a": 0.2}}, {"label": "a=0.05", "values": {"a": 0.05}}],
    "Ch5 · Add Rayleigh noise": [{"label": "b=50", "values": {"b": 50}}, {"label": "b=800", "values": {"b": 800}}],
    "Ch5 · Add Poisson noise": [{"label": "µ=10", "values": {"mu": 10}}, {"label": "µ=150", "values": {"mu": 150}}],
    "Ch5 · Arithmetic mean": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch5 · Geometric mean": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch5 · Harmonic mean": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch5 · Contraharmonic mean": [
        {"label": "Q=1.5 (remove pepper)", "values": {"k": 9, "Q": 1.5}},
        {"label": "Q=−1.5 (remove salt)", "values": {"k": 9, "Q": -1.5}},
    ],
    "Ch5 · Median": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch5 · Min": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch5 · Max": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch5 · Midpoint": [{"label": "k=3", "values": {"k": 3}}, {"label": "k=15", "values": {"k": 15}}],
    "Ch5 · Alpha-trimmed": [{"label": "k=3", "values": {"k": 3, "alpha": 0.25}}, {"label": "k=15", "values": {"k": 15, "alpha": 0.25}}],
}


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

KERNEL  = {"name": "k", "min": 3, "max": 15, "default": 9, "step": 2, "int": True}
GKERNEL = {"name": "k", "min": 3, "max": 31, "default": 9, "step": 2, "int": True}
SIGMA   = {"name": "sigma", "min": 0.5, "max": 10, "default": 2.0, "step": 0.1}
CUTOFF  = {"name": "D_0", "min": 0.01, "max": 0.5, "default": 0.1, "step": 0.01}
ORDER   = {"name": "n", "min": 1, "max": 10, "default": 2, "step": 1, "int": True}
SEED    = {"name": "seed", "min": 0, "max": 99, "default": 0, "step": 1, "int": True}

FILTERS = {
    # Ch2 - resolution & depth (handout exercises)
    "Ch2 · Downsample (resolution)":   {"fn": downsample, "params": [{"name": "f", "min": 2, "max": 16, "default": 4, "step": 2, "int": True}], "formula": r"g = \text{nearest}(f):\quad N/f \times N/f \to N \times N"},
    "Ch2 · Bit-depth reduction":       {"fn": bit_depth_reduce, "params": [{"name": "b", "min": 1, "max": 7, "default": 4, "step": 1, "int": True}], "formula": r"s = \operatorname{round}\left(\frac{r}{255}(2^b-1)\right)\frac{255}{2^b-1}"},
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
    "Ch5 · Add Gaussian noise":    {"fn": add_gaussian_noise, "params": [{"name": "sigma", "min": 1, "max": 100, "default": 20, "step": 1, "int": True}, SEED], "formula": r"g = f + \sigma\,\varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,1)"},
    "Ch5 · Add salt-and-pepper":   {"fn": add_salt_pepper, "params": [{"name": "p", "min": 0.01, "max": 0.5, "default": 0.05, "step": 0.01}, SEED], "formula": r"g = \begin{cases} 0 & \text{pepper (prob } p/2\text{)} \\ 255 & \text{salt (prob } p/2\text{)} \\ f & \text{otherwise} \end{cases}"},
    "Ch5 · Add uniform noise":      {"fn": add_uniform_noise, "params": [{"name": "A", "min": 1, "max": 100, "default": 30, "step": 1, "int": True}, SEED], "formula": r"p(z) = \frac{1}{2A},\ z \in [-A, A],\quad g = f + z"},
    "Ch5 · Add Erlang (gamma) noise": {"fn": add_erlang_noise, "params": [
        {"name": "a", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
        {"name": "b", "min": 1, "max": 10, "default": 2, "step": 1, "int": True},
        SEED,
    ], "formula": r"p(z) = \frac{a^b z^{b-1} e^{-az}}{(b-1)!},\ z \ge 0,\quad g = f + z - \frac{b}{a}"},
    "Ch5 · Add exponential noise":  {"fn": add_exponential_noise, "params": [{"name": "a", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01}, SEED], "formula": r"p(z) = a e^{-az},\ z \ge 0,\quad g = f + z - \frac{1}{a}"},
    "Ch5 · Add Rayleigh noise":     {"fn": add_rayleigh_noise, "params": [{"name": "b", "min": 10, "max": 2000, "default": 200, "step": 10, "int": True}, SEED], "formula": r"p(z) = \frac{2}{b}(z-a) e^{-(z-a)^2/b},\ z \ge a,\quad g = f + z - \tfrac{1}{2}\sqrt{\pi b}"},
    "Ch5 · Add Poisson noise":      {"fn": add_poisson_noise, "params": [{"name": "mu", "min": 1, "max": 200, "default": 50, "step": 1, "int": True}, SEED], "formula": r"p(z) = \frac{e^{-\mu} \mu^z}{z!},\ z \ge 0,\quad g = f + z - \mu"},
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

    # Ch2 - resolution & depth
    big = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert downsample(big, 2).shape == (8, 8)
    assert downsample(big, 2)[1, 1] == big[0, 0]        # nearest-neighbor blocks
    assert downsample(big, 4)[3, 3] == big[0, 0]
    assert downsample(big, 6).shape == (8, 8)           # non-divisor f keeps size
    assert downsample(big, 6)[0, 0] == big[0, 0] and downsample(big, 6)[7, 7] == big[6, 6]
    assert bit_depth_reduce(np.full((3, 3), 100, np.uint8), 8).max() == 100  # b=8 -> identity
    assert bit_depth_reduce(np.full((3, 3), 200, np.uint8), 1)[0, 0] == 255  # b=1 -> binary
    assert bit_depth_reduce(np.full((3, 3), 100, np.uint8), 1)[0, 0] == 0

    # new noise types: zero-mean, stats within tolerance
    np.random.seed(0)
    base = np.full((200, 200), 128, np.uint8)
    assert np.abs(add_uniform_noise(base, 50).astype(float) - 128).max() <= 50
    assert np.abs(add_erlang_noise(base, 0.1, 2).astype(float).mean() - 128) < 2
    assert np.abs(add_exponential_noise(base, 0.1).astype(float).mean() - 128) < 2
    assert np.abs(add_rayleigh_noise(base, 200).astype(float).mean() - 128) < 2
    assert np.abs(add_poisson_noise(base, 50).astype(float).mean() - 128) < 2

    # new metrics + entropy
    a = np.arange(16, dtype=float).reshape(4, 4); b = a + 10
    assert me(a, b) == 10.0
    assert abs(nmse(a, b) - np.sum((a - b) ** 2) / np.sum(a ** 2)) < 1e-9
    assert abs(covariance(a, b) - np.mean((a - a.mean()) * (b - b.mean()))) < 1e-9
    assert abs(correlation(a, b) - 1.0) < 1e-9            # perfect linear -> rho = 1
    assert jaccard(a, a) == 1.0
    assert jaccard(a, b) == 0.0
    assert entropy(np.full((8, 8), 5, np.uint8)) == 0.0    # constant -> 0 bits
    assert abs(entropy(np.arange(256, dtype=np.uint8).reshape(16, 16)) - 8.0) < 1e-6  # uniform histogram -> 8 bits

    assert all(f.get("formula") for f in FILTERS.values())
    assert set(CAPTIONS) == set(FILTERS), "every filter needs a caption"
    for fname, presets in PRESETS.items():
        assert fname in FILTERS, fname
        names = {p["name"] for p in FILTERS[fname]["params"]}
        for pr in presets:
            assert pr["label"] and set(pr["values"]) <= names, (fname, pr)
            for k, v in pr["values"].items():
                p = next(x for x in FILTERS[fname]["params"] if x["name"] == k)
                assert p["min"] <= v <= p["max"], (fname, k, v)
    print(f"filters OK ({len(FILTERS)} filters)")


if __name__ == "__main__":
    demo()
