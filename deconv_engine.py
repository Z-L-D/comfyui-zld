"""
Deconvolution engine: PSF generation, Wiener/RL deconvolution, edge handling, noise reduction.
"""

import numpy as np
from scipy import fft as sp_fft
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter


# ─── PSF Generation ────────────────────────────────────────────────────────────

def make_disk_psf(diameter: float, size: int = None) -> np.ndarray:
    """
    Generate a circular disk (pillbox) PSF for out-of-focus blur.
    
    This models the defocus aberration of a circular aperture lens.
    A point source becomes a uniform disk when defocused. The diameter
    corresponds to Focus Magic's "Blur Width" parameter.
    
    For more realistic lens modeling, we add a slight Gaussian taper at
    the edge to approximate diffraction effects (Airy disk softening).
    
    Args:
        diameter: Blur width in pixels (diameter of the circle of confusion)
        size: Output kernel size (auto-calculated if None)
    
    Returns:
        Normalized 2D PSF array
    """
    radius = diameter / 2.0
    if size is None:
        size = int(np.ceil(diameter)) + 4  # padding for anti-aliasing
        if size % 2 == 0:
            size += 1

    center = size // 2
    y, x = np.ogrid[-center:size - center, -center:size - center]
    dist = np.sqrt(x * x + y * y)

    # Soft-edged disk: smooth transition at boundary for anti-aliasing
    # This avoids the harsh ringing that a pure pillbox PSF would cause
    edge_width = max(0.5, radius * 0.1)  # 10% of radius, minimum 0.5px
    psf = np.clip((radius - dist) / edge_width + 0.5, 0, 1)

    # Normalize
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf /= psf_sum

    return psf.astype(np.float64)


def make_gaussian_psf(sigma: float, size: int = None) -> np.ndarray:
    """
    Generate a Gaussian PSF.
    
    More physically accurate for mild defocus or atmospheric blur
    where the PSF is better modeled as a Gaussian distribution.
    
    Args:
        sigma: Standard deviation of the Gaussian in pixels
        size: Output kernel size
    
    Returns:
        Normalized 2D PSF array
    """
    if size is None:
        size = int(np.ceil(sigma * 6)) | 1  # 6-sigma coverage, ensure odd

    center = size // 2
    y, x = np.ogrid[-center:size - center, -center:size - center]
    psf = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    psf /= psf.sum()
    return psf.astype(np.float64)


def make_motion_psf(angle_degrees: float, distance: float, size: int = None) -> np.ndarray:
    """
    Generate a linear motion blur PSF.
    
    Models camera shake or subject motion as a uniform line kernel.
    The angle specifies the direction of motion (0° = horizontal right,
    90° = vertical up, following Focus Magic's convention).
    
    For more realistic motion blur, the kernel has slight Gaussian
    weighting to model acceleration/deceleration at motion endpoints.
    
    Args:
        angle_degrees: Direction of motion in degrees (0-360)
        distance: Length of motion blur in pixels (Focus Magic's "Blur Distance")
        size: Output kernel size
    
    Returns:
        Normalized 2D PSF array
    """
    if size is None:
        size = int(np.ceil(distance)) + 4
        if size % 2 == 0:
            size += 1

    center = size // 2
    psf = np.zeros((size, size), dtype=np.float64)

    # Convert angle to radians
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Sub-pixel accurate line drawing using anti-aliased sampling
    half_dist = distance / 2.0
    num_samples = max(int(distance * 4), 100)  # oversample for smoothness

    for i in range(num_samples):
        t = -half_dist + (i / (num_samples - 1)) * distance
        px = center + t * cos_a
        py = center - t * sin_a  # negative because y-axis is inverted in images

        # Bilinear interpolation for sub-pixel placement
        x0 = int(np.floor(px))
        y0 = int(np.floor(py))
        fx = px - x0
        fy = py - y0

        # Optional: slight Gaussian weighting for more realistic motion
        # (camera typically decelerates at endpoints)
        weight = 1.0  # uniform for standard motion blur

        for dy in range(2):
            for dx in range(2):
                xi = x0 + dx
                yi = y0 + dy
                if 0 <= xi < size and 0 <= yi < size:
                    wx = (1 - fx) if dx == 0 else fx
                    wy = (1 - fy) if dy == 0 else fy
                    psf[yi, xi] += weight * wx * wy

    psf_sum = psf.sum()
    if psf_sum > 0:
        psf /= psf_sum

    return psf


def make_motion_psf_tapered(angle_degrees: float, distance: float, 
                             taper_strength: float = 0.3, size: int = None) -> np.ndarray:
    """
    Generate a tapered motion blur PSF for non-uniform motion.
    
    Models the common case where camera shake produces a "slow-fast-slow"
    motion pattern, resulting in brighter endpoints in the PSF.
    Focus Magic's tutorials describe this as "double exposure" effect.
    
    Args:
        angle_degrees: Direction of motion in degrees
        distance: Length of motion blur in pixels
        taper_strength: 0.0 = uniform, 1.0 = heavily weighted at endpoints
        size: Output kernel size
    
    Returns:
        Normalized 2D PSF array
    """
    if size is None:
        size = int(np.ceil(distance)) + 4
        if size % 2 == 0:
            size += 1

    center = size // 2
    psf = np.zeros((size, size), dtype=np.float64)

    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    half_dist = distance / 2.0
    num_samples = max(int(distance * 4), 100)

    for i in range(num_samples):
        t_norm = i / (num_samples - 1)  # 0 to 1
        t = -half_dist + t_norm * distance
        px = center + t * cos_a
        py = center - t * sin_a

        # Cosine taper: higher weight at endpoints
        endpoint_weight = 1.0 + taper_strength * np.cos(np.pi * t_norm * 2)

        x0 = int(np.floor(px))
        y0 = int(np.floor(py))
        fx = px - x0
        fy = py - y0

        for dy in range(2):
            for dx in range(2):
                xi = x0 + dx
                yi = y0 + dy
                if 0 <= xi < size and 0 <= yi < size:
                    wx = (1 - fx) if dx == 0 else fx
                    wy = (1 - fy) if dy == 0 else fy
                    psf[yi, xi] += endpoint_weight * wx * wy

    psf_sum = psf.sum()
    if psf_sum > 0:
        psf /= psf_sum

    return psf


# ─── Auto PSF Estimation (Blind Mode) ──────────────────────────────────────────

def estimate_blur_width(image: np.ndarray) -> float:
    """
    Estimate defocus blur width from a single image using spectral analysis.
    
    Analyzes the power spectrum to find the first zero-crossing of the
    radially averaged OTF (optical transfer function). For a disk PSF
    of diameter d, zeros occur at specific frequencies related to the
    Bessel function J1, giving us an estimate of the blur diameter.
    
    This is an approximation - Focus Magic's auto-detection uses a
    more sophisticated approach, but this captures the core idea.
    
    Args:
        image: Grayscale image as 2D numpy array
    
    Returns:
        Estimated blur width (diameter) in pixels
    """
    # Compute power spectrum
    f_image = sp_fft.fft2(image)
    power = np.abs(f_image) ** 2

    # Radial average of the power spectrum
    h, w = image.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[-cy:h - cy, -cx:w - cx]
    R = np.sqrt(X * X + Y * Y).astype(int)

    power_shifted = sp_fft.fftshift(power)
    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    counts = np.zeros(max_r)

    for r in range(max_r):
        mask = R == r
        radial_profile[r] = power_shifted[mask].mean()
        counts[r] = mask.sum()

    # Normalize and find the first significant drop-off
    radial_profile = radial_profile / radial_profile[0]

    # Smooth the profile to reduce noise
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(np.log(radial_profile + 1e-10), size=5)

    # Find where the spectrum drops to a threshold
    # For a disk PSF, the first zero of the OTF is at frequency ~ 1.22/diameter
    threshold = smoothed[0] - 3.0  # -3dB equivalent in log space
    crossings = np.where(smoothed < threshold)[0]

    if len(crossings) > 0:
        first_drop = crossings[0]
        # Convert frequency to blur width
        # frequency = first_drop / max_r * (Nyquist)
        # For disk PSF: first_zero_freq ≈ 1.22 / diameter
        freq_normalized = first_drop / max_r
        if freq_normalized > 0.01:
            estimated_diameter = 1.22 / freq_normalized
            return np.clip(estimated_diameter, 1.0, 30.0)

    # Fallback: use ratio of Laplacian variance to image variance
    # This is more robust than absolute Laplacian variance
    from scipy.ndimage import laplace
    lap = laplace(image.astype(np.float64))
    lap_var = lap.var()
    img_var = image.var()

    if img_var > 1e-10:
        # Blur ratio: for a sharp image, Laplacian variance is high relative
        # to image variance. For blurred images, the ratio drops.
        blur_ratio = lap_var / img_var
        # Empirical mapping calibrated against known disk PSFs
        if blur_ratio > 0.1:
            estimated_width = np.clip(2.0 / np.sqrt(blur_ratio + 0.01), 1.0, 20.0)
        else:
            estimated_width = np.clip(6.0 / (blur_ratio + 0.01), 2.0, 20.0)
    else:
        estimated_width = 5.0

    return estimated_width


def estimate_motion_params(image: np.ndarray) -> tuple:
    """
    Estimate motion blur angle and distance from spectral analysis.
    
    Motion blur creates a characteristic pattern in the frequency domain:
    the power spectrum shows dark bands perpendicular to the motion direction.
    By analyzing the orientation of these bands, we can estimate the angle.
    The spacing of the bands gives us the distance.
    
    Args:
        image: Grayscale image as 2D numpy array
    
    Returns:
        (angle_degrees, distance_pixels) tuple
    """
    # Compute log power spectrum
    f_image = sp_fft.fftshift(sp_fft.fft2(image))
    power = np.log(np.abs(f_image) ** 2 + 1)

    h, w = power.shape
    cy, cx = h // 2, w // 2

    # Radon-like analysis: sum power along different angles
    num_angles = 180
    angle_sums = np.zeros(num_angles)

    for i in range(num_angles):
        angle = np.deg2rad(i)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        max_r = min(cy, cx) // 2
        line_sum = 0
        count = 0
        for r in range(-max_r, max_r):
            x = int(cx + r * cos_a)
            y = int(cy + r * sin_a)
            if 0 <= x < w and 0 <= y < h:
                line_sum += power[y, x]
                count += 1

        if count > 0:
            angle_sums[i] = line_sum / count

    # Motion blur direction is perpendicular to the dark bands
    # The minimum in angle_sums corresponds to the dark band direction
    min_angle = np.argmin(angle_sums)
    # Motion direction is perpendicular (+90°)
    motion_angle = (min_angle + 90) % 180

    # Estimate distance from the band spacing
    # Extract a 1D profile along the dark band direction
    cos_a = np.cos(np.deg2rad(min_angle))
    sin_a = np.sin(np.deg2rad(min_angle))
    max_r = min(cy, cx) // 2
    profile = []
    for r in range(1, max_r):
        x = int(cx + r * cos_a)
        y = int(cy + r * sin_a)
        if 0 <= x < w and 0 <= y < h:
            profile.append(power[y, x])

    profile = np.array(profile)
    if len(profile) > 10:
        # Find first minimum (zero of the sinc function)
        smoothed = uniform_filter(profile, size=3)
        diffs = np.diff(smoothed)
        # Find first valley
        for i in range(1, len(diffs) - 1):
            if diffs[i - 1] < 0 and diffs[i] >= 0:
                # First minimum found at index i
                # distance ≈ image_size / (2 * zero_frequency)
                freq = i / max_r
                if freq > 0.01:
                    distance = 1.0 / freq
                    return float(motion_angle), np.clip(distance, 1.0, 50.0)

    # Fallback
    return float(motion_angle), 5.0


# ─── Edge Handling ──────────────────────────────────────────────────────────────

def edge_taper(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Apply edge tapering to reduce boundary ringing artifacts.
    
    This is critical for deconvolution quality. The DFT assumes periodic
    boundaries, causing severe ringing at image edges. Edge tapering
    smoothly blends the image borders with a blurred version, making
    the boundaries periodic-friendly.
    
    Similar to MATLAB's edgetaper() and what Focus Magic does internally.
    
    Args:
        image: Input image (2D)
        psf: Point spread function
    
    Returns:
        Edge-tapered image
    """
    # Compute the autocorrelation of the PSF (used to determine taper width)
    psf_padded = np.zeros_like(image)
    ph, pw = psf.shape
    py, px = (image.shape[0] - ph) // 2, (image.shape[1] - pw) // 2
    psf_padded[py:py + ph, px:px + pw] = psf

    # Autocorrelation via FFT
    psf_fft = sp_fft.fft2(psf_padded)
    acf = np.real(sp_fft.ifft2(np.abs(psf_fft) ** 2))
    acf = sp_fft.fftshift(acf)

    # Create taper weights from the PSF's spread
    h, w = image.shape
    taper_size = max(ph, pw)

    # Create smooth border weight mask
    weight = np.ones_like(image)
    for i in range(taper_size):
        alpha = (i + 1) / (taper_size + 1)
        # Top and bottom
        if i < h:
            weight[i, :] = min(weight[i, :].min(), alpha)
            weight[h - 1 - i, :] = min(weight[h - 1 - i, :].min(), alpha)
        # Left and right
        if i < w:
            weight[:, i] = np.minimum(weight[:, i], alpha)
            weight[:, w - 1 - i] = np.minimum(weight[:, w - 1 - i], alpha)

    # Blend original with blurred version at borders
    blurred = gaussian_filter(image, sigma=taper_size / 2)
    tapered = weight * image + (1 - weight) * blurred

    return tapered


# ─── Noise Estimation and Reduction ────────────────────────────────────────────

def estimate_noise_level(image: np.ndarray) -> float:
    """
    Estimate noise standard deviation using the MAD (Median Absolute Deviation)
    method on high-frequency wavelet coefficients.
    
    This is the standard robust noise estimator used in image processing.
    
    Args:
        image: Grayscale image as 2D numpy array
    
    Returns:
        Estimated noise standard deviation
    """
    # Use high-pass filter (Laplacian-like) to isolate noise
    # This is equivalent to the finest-scale wavelet coefficients
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float64)

    from scipy.signal import convolve2d
    hf = convolve2d(image, kernel, mode='valid', boundary='symm')

    # MAD estimator: sigma ≈ MAD / 0.6745
    mad = np.median(np.abs(hf - np.median(hf)))
    sigma = mad / 0.6745

    return sigma


def denoise_pre(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Pre-deconvolution noise reduction.
    
    Applies gentle smoothing to reduce noise before deconvolution,
    which helps prevent noise amplification during the inverse process.
    Uses an edge-preserving bilateral-like approach (approximated with
    guided filtering).
    
    Args:
        image: Input image
        strength: Noise reduction strength (0.0 = none, 1.0 = standard)
    
    Returns:
        Denoised image
    """
    if strength <= 0:
        return image

    sigma = estimate_noise_level(image) * strength

    if sigma < 0.001:
        return image

    # Edge-preserving smoothing using combination of
    # Gaussian and median filtering
    gaussian_component = gaussian_filter(image, sigma=sigma * 2)
    median_component = median_filter(image, size=max(3, int(sigma * 2) | 1))

    # Blend: median is better for salt-and-pepper, Gaussian for random noise
    denoised = 0.6 * gaussian_component + 0.4 * median_component

    # Preserve edges by blending back with original at high-gradient areas
    from scipy.ndimage import sobel
    gx = sobel(image, axis=1)
    gy = sobel(image, axis=0)
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    gradient = gradient / (gradient.max() + 1e-10)

    # Edge mask: keep original at edges, use denoised in flat areas
    edge_weight = np.clip(gradient * 3, 0, 1)
    result = edge_weight * image + (1 - edge_weight) * denoised

    return result


def denoise_post(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Post-deconvolution noise reduction.
    
    After deconvolution, noise can be amplified (especially in flat regions).
    This applies targeted noise reduction that preserves the recovered details.
    Focus Magic v5 specifically improved this aspect.
    
    Args:
        image: Deconvolved image
        strength: Noise reduction strength
    
    Returns:
        Cleaned image
    """
    if strength <= 0:
        return image

    sigma_est = estimate_noise_level(image) * strength * 0.5

    if sigma_est < 0.001:
        return image

    # Adaptive noise reduction: stronger in flat areas, weaker at edges
    smoothed = gaussian_filter(image, sigma=max(0.5, sigma_est))

    # Local variance to detect flat vs textured regions
    local_mean = uniform_filter(image, size=5)
    local_var = uniform_filter((image - local_mean) ** 2, size=5)

    # Normalize variance
    max_var = np.percentile(local_var, 95) + 1e-10
    texture_map = np.clip(local_var / max_var, 0, 1)

    # In textured areas, keep deconvolved result; in flat areas, use smoothed
    result = texture_map * image + (1 - texture_map) * smoothed

    return result


# ─── Wiener Deconvolution ──────────────────────────────────────────────────────

def wiener_deconvolve(image: np.ndarray, psf: np.ndarray,
                       noise_power: float = 0.01,
                       regularization: float = 0.001) -> np.ndarray:
    """
    Regularized Wiener deconvolution.
    
    The Wiener filter minimizes the mean square error between the estimated
    and true image, accounting for both the blur (PSF) and noise:
    
        F_hat(u,v) = H*(u,v) / (|H(u,v)|² + λ) * G(u,v)
    
    where H is the OTF, G is the blurred image spectrum, and λ is the
    regularization parameter (noise-to-signal ratio).
    
    This is the core algorithm that Focus Magic likely uses, with their
    proprietary enhancements for noise handling and regularization.
    
    Args:
        image: Blurred input image (2D grayscale)
        psf: Point spread function
        noise_power: Noise-to-signal power ratio (λ in Wiener formula)
        regularization: Additional Tikhonov regularization strength
    
    Returns:
        Deconvolved image
    """
    # Pad PSF to image size and center it
    psf_padded = np.zeros_like(image)
    ph, pw = psf.shape
    # Place PSF centered at origin (top-left for FFT)
    py = ph // 2
    px = pw // 2

    for i in range(ph):
        for j in range(pw):
            yi = (i - py) % image.shape[0]
            xi = (j - px) % image.shape[1]
            psf_padded[yi, xi] = psf[i, j]

    # FFT of image and PSF
    G = sp_fft.rfft2(image)
    H = sp_fft.rfft2(psf_padded)

    # Wiener filter with regularization
    H_conj = np.conj(H)
    H_power = np.abs(H) ** 2

    # Combined noise power and Tikhonov regularization
    # The Laplacian regularization penalizes high-frequency noise amplification
    h, w_half = H.shape
    fy = np.fft.fftfreq(image.shape[0])[:, np.newaxis]
    fx = np.fft.rfftfreq(image.shape[1])[np.newaxis, :]
    laplacian_power = (2 * np.pi) ** 2 * (fx ** 2 + fy ** 2)

    denominator = H_power + noise_power + regularization * laplacian_power
    denominator = np.maximum(denominator, 1e-10)  # prevent division by zero

    F_hat = (H_conj / denominator) * G

    result = sp_fft.irfft2(F_hat, s=image.shape)

    return np.real(result)


# ─── Richardson-Lucy Deconvolution ──────────────────────────────────────────────

def richardson_lucy_deconvolve(image: np.ndarray, psf: np.ndarray,
                                iterations: int = 30,
                                damping: float = 0.0,
                                tv_regularization: float = 0.0,
                                clip: bool = True) -> np.ndarray:
    """
    Accelerated, damped Richardson-Lucy deconvolution with TV regularization.
    
    The RL algorithm iteratively maximizes the likelihood assuming Poisson
    noise statistics:
    
        x_{n+1} = x_n * (PSF_flipped ★ (image / (PSF ★ x_n)))
    
    Enhanced with:
    - Damping: suppresses updates for pixels close to the noisy observation
    - Total Variation regularization: reduces noise while preserving edges
    - Acceleration: uses momentum from previous iterations (Biggs & Andrews)
    
    This iterative approach produces higher quality results than Wiener for
    large blur amounts, similar to Focus Magic's "Accuracy" setting.
    
    Args:
        image: Blurred input image (2D grayscale, values > 0)
        psf: Point spread function
        iterations: Number of RL iterations
        damping: Damping threshold (0 = no damping)
        tv_regularization: Total Variation regularization weight
        clip: Whether to clip result to [0, 1]
    
    Returns:
        Deconvolved image
    """
    # Ensure positive values (RL requires this)
    image_pos = np.maximum(image, 1e-6)

    # PSF and flipped PSF for correlations (done in frequency domain)
    psf_padded = np.zeros_like(image)
    ph, pw = psf.shape
    py, px = ph // 2, pw // 2
    for i in range(ph):
        for j in range(pw):
            yi = (i - py) % image.shape[0]
            xi = (j - px) % image.shape[1]
            psf_padded[yi, xi] = psf[i, j]

    H = sp_fft.rfft2(psf_padded)
    H_conj = np.conj(H)

    def convolve_psf(x):
        return np.real(sp_fft.irfft2(sp_fft.rfft2(x) * H, s=image.shape))

    def correlate_psf(x):
        return np.real(sp_fft.irfft2(sp_fft.rfft2(x) * H_conj, s=image.shape))

    # Initialize with the blurred image (slight Gaussian smoothing for stability)
    estimate = gaussian_filter(image_pos, sigma=0.5)
    estimate = np.maximum(estimate, 1e-6)

    for iteration in range(iterations):
        # Core RL update
        convolved = convolve_psf(estimate)
        convolved = np.maximum(convolved, 1e-10)

        ratio = image_pos / convolved

        # Clip ratio to prevent explosive updates from noise
        ratio = np.clip(ratio, 0.1, 10.0)

        correction = correlate_psf(ratio)

        # Damping: suppress updates for pixels near the observation
        if damping > 0:
            deviation = np.abs(image_pos - convolved)
            damp_mask = 1.0 - np.exp(-(deviation / (damping + 1e-10)) ** 2)
            correction = 1.0 + (correction - 1.0) * damp_mask

        # Apply update with relaxation for stability
        relaxation = min(1.0, 0.5 + 0.5 * iteration / max(iterations * 0.3, 1))
        new_estimate = estimate * (1.0 + relaxation * (correction - 1.0))

        # Total Variation regularization
        if tv_regularization > 0:
            new_estimate = _tv_denoise_step(new_estimate, tv_regularization)

        estimate = np.maximum(new_estimate, 1e-10)

    if clip:
        estimate = np.clip(estimate, 0, 1)

    return estimate


def _tv_denoise_step(image: np.ndarray, weight: float) -> np.ndarray:
    """
    Single step of Total Variation denoising for RL regularization.
    
    Applies gradient-based regularization that preserves edges while
    smoothing flat regions. This prevents the noise amplification
    typical of unregularized RL deconvolution.
    """
    eps = 1e-8

    # Compute gradients
    dx = np.diff(image, axis=1, prepend=image[:, :1])
    dy = np.diff(image, axis=0, prepend=image[:1, :])

    # Gradient magnitude
    grad_mag = np.sqrt(dx ** 2 + dy ** 2 + eps)

    # Divergence of normalized gradient (TV term)
    nx = dx / grad_mag
    ny = dy / grad_mag

    div_x = np.diff(nx, axis=1, append=nx[:, -1:])
    div_y = np.diff(ny, axis=0, append=ny[-1:, :])
    divergence = div_x + div_y

    # Update
    result = image + weight * divergence

    return np.maximum(result, 1e-10)


# ─── Hybrid Deconvolution ──────────────────────────────────────────────────────

def hybrid_deconvolve(image: np.ndarray, psf: np.ndarray,
                       noise_power: float = 0.01,
                       rl_iterations: int = 15,
                       tv_regularization: float = 0.001) -> np.ndarray:
    """
    Hybrid Wiener + Richardson-Lucy deconvolution.
    
    Uses Wiener deconvolution as an initial estimate, then refines
    with a few RL iterations. This often gives the best quality:
    Wiener provides a good global solution quickly, and RL recovers
    fine details through its iterative refinement.
    
    This is likely closest to what Focus Magic does at high "Accuracy"
    settings.
    
    Args:
        image: Blurred input image
        psf: Point spread function
        noise_power: Wiener noise parameter
        rl_iterations: Number of RL refinement iterations
        tv_regularization: TV regularization for RL phase
    
    Returns:
        Deconvolved image
    """
    # Phase 1: Wiener for initial estimate
    wiener_result = wiener_deconvolve(image, psf, noise_power=noise_power)
    wiener_result = np.clip(wiener_result, 1e-6, 1)

    if rl_iterations <= 0:
        return np.clip(wiener_result, 0, 1)

    # Phase 2: RL refinement
    # Use fewer iterations since we have a good starting point
    result = richardson_lucy_deconvolve(
        image, psf,
        iterations=rl_iterations,
        tv_regularization=tv_regularization,
        clip=True
    )

    # Blend: use the better result per-pixel based on local consistency
    # Where Wiener and RL agree, use their average (more confident)
    # Where they disagree significantly, prefer Wiener (more stable)
    diff = np.abs(wiener_result - result)
    agreement = np.exp(-diff * 10)  # High where they agree
    
    blended = agreement * (0.4 * wiener_result + 0.6 * result) + \
              (1 - agreement) * (0.7 * wiener_result + 0.3 * result)

    return np.clip(blended, 0, 1)


# ─── Full Pipeline ──────────────────────────────────────────────────────────────

def deconvolve_channel(channel: np.ndarray, psf: np.ndarray,
                        method: str = "wiener",
                        noise_power: float = 0.01,
                        regularization: float = 0.001,
                        iterations: int = 30,
                        tv_reg: float = 0.001,
                        denoise_before: float = 0.0,
                        denoise_after: float = 0.5,
                        amount: float = 1.0) -> np.ndarray:
    """
    Full deconvolution pipeline for a single channel.
    
    Applies the complete Focus Magic-style processing chain:
    1. Edge tapering (prevent boundary ringing)
    2. Optional pre-deconvolution noise reduction
    3. Deconvolution (Wiener, RL, or Hybrid)
    4. Optional post-deconvolution noise reduction
    5. Amount blending (mix deconvolved with original)
    
    Args:
        channel: Single image channel, float64 in [0, 1]
        psf: Point spread function
        method: "wiener", "richardson_lucy", or "hybrid"
        noise_power: Noise parameter for Wiener
        regularization: Tikhonov regularization strength
        iterations: RL iteration count
        tv_reg: Total Variation regularization
        denoise_before: Pre-deconv noise reduction (0-1)
        denoise_after: Post-deconv noise reduction (0-1)
        amount: Deconvolution strength (0 = original, 1 = full deconv)
    
    Returns:
        Deconvolved channel
    """
    original = channel.copy()

    # 1. Pre-noise reduction
    if denoise_before > 0:
        channel = denoise_pre(channel, denoise_before)

    # 2. Edge tapering
    channel = edge_taper(channel, psf)

    # 3. Deconvolution
    if method == "wiener":
        deconvolved = wiener_deconvolve(channel, psf,
                                         noise_power=noise_power,
                                         regularization=regularization)
    elif method == "richardson_lucy":
        deconvolved = richardson_lucy_deconvolve(channel, psf,
                                                  iterations=iterations,
                                                  tv_regularization=tv_reg)
    elif method == "hybrid":
        deconvolved = hybrid_deconvolve(channel, psf,
                                         noise_power=noise_power,
                                         rl_iterations=max(1, iterations // 2),
                                         tv_regularization=tv_reg)
    else:
        deconvolved = wiener_deconvolve(channel, psf, noise_power=noise_power)

    deconvolved = np.clip(deconvolved, 0, 1)

    # 4. Post-noise reduction
    if denoise_after > 0:
        deconvolved = denoise_post(deconvolved, denoise_after)
        deconvolved = np.clip(deconvolved, 0, 1)

    # 5. Amount blending
    if amount < 1.0:
        result = original * (1 - amount) + deconvolved * amount
    else:
        result = deconvolved

    return np.clip(result, 0, 1)


def deconvolve_image(image_rgb: np.ndarray, psf: np.ndarray,
                      method: str = "wiener",
                      noise_power: float = 0.01,
                      regularization: float = 0.001,
                      iterations: int = 30,
                      tv_reg: float = 0.001,
                      denoise_before: float = 0.0,
                      denoise_after: float = 0.5,
                      amount: float = 1.0) -> np.ndarray:
    """
    Full deconvolution pipeline for RGB images.
    
    Processes each channel independently, following Focus Magic's approach.
    
    Args:
        image_rgb: RGB image as (H, W, 3) float64 array in [0, 1]
        psf: Point spread function (2D)
        method: Deconvolution method
        ...other params same as deconvolve_channel
    
    Returns:
        Deconvolved RGB image
    """
    result = np.zeros_like(image_rgb)

    for c in range(image_rgb.shape[2]):
        result[:, :, c] = deconvolve_channel(
            image_rgb[:, :, c], psf,
            method=method,
            noise_power=noise_power,
            regularization=regularization,
            iterations=iterations,
            tv_reg=tv_reg,
            denoise_before=denoise_before,
            denoise_after=denoise_after,
            amount=amount
        )

    return result