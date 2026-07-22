import numpy as np
from scipy.ndimage import maximum_filter,minimum_filter
from scipy.fft import rfft2, irfft2
from scipy.signal import tukey


def _make_tukey_window(shape, feather=0.1):
    rows, cols = shape
    wy = tukey(rows, feather, sym=True)
    wx = tukey(cols, feather, sym=True)
    window2d = np.outer(wy, wx)
    return window2d



def wavelet_image_quality_metric(
    image,
    filter_bank=None, 
    coarse_neighborhood_size=3,
    denoise_strength=1.0,
    product_mode = 'signed_coefficients',
    filter_bank_params=None,
):
    """
    Wavelets...
    """
    
    #Load settings
    defaults = dict(
        #frequency_levels=[0.75, 0.375, 0.1875, 0.09375, 0.046875],
        frequency_levels=[0.5, 0.45, 0.4, 0.35, 0.3],
        cwt_bank=None,
        pad=True, 
        pad_mode="reflect", 
        pad_factor=6.0,
        wavelet_bank_normalization="unit_energy",
    )
    if filter_bank_params is not None:
        defaults.update(filter_bank_params)
    p = defaults

    image = np.asarray(image, dtype=np.float64)
    work_image = _anscombe_transform(image)   # Poisson -> ~unit-variance Gaussian

    bank = filter_bank
    if bank is None:
        if filter_bank_params is None:
            raise RuntimeError('Please provide either a prebuilt wavelet filter bank or a set of parameters to build one')
        else:
            bank = make_isotropic_cwt2d_mexican_hat_bank(
                image_shape=image.shape,
                center_frequency_fractions=p["frequency_levels"],
                pixel_size_um=p['pixel_size_um'],
                mode=p['mode'],
                NA=p['NA'],
                wavelength_em=p['wavelength_em'],
                wavelength_ex=p['wavelength_ex'],
                pad=p["pad"], 
                pad_mode=p["pad_mode"], 
                pad_factor=p["pad_factor"],
                normalization=p["wavelet_bank_normalization"])
    else:
        if bank["image_shape"] != image.shape:
            raise ValueError("bank/image shape mismatch")

    coeffs = _isotropic_cwt2d_mexican_hat_with_bank(image=work_image, bank=bank, pad_mode=p["pad_mode"])

    for level in range(coeffs.shape[0]):
        sigma = _estimate_sigma_mad(coeffs[level])
        coeffs[level] = _soft_threshold(coeffs[level], denoise_strength * sigma)

    energies = []
    metric = 0.0
    if product_mode == 'absolute_values':
        for i in range(coeffs.shape[0] - 1):
            fine = np.abs(coeffs[i])
            coarse_max = maximum_filter(np.abs(coeffs[i + 1]),size=coarse_neighborhood_size)
            tukey_window = _make_tukey_window(fine.shape)
            E = np.sum(tukey_window * fine * coarse_max)
            E_floor = np.sum(tukey_window) * (np.sum(fine*tukey_window)/np.sum(tukey_window)) * (np.sum(coarse_max*tukey_window)/np.sum(tukey_window)) 
            energies.append(max(E-E_floor,0.0))
            metric +=max(E-E_floor,0.0)

    elif product_mode == 'signed_coefficients':
        for i in range(coeffs.shape[0] - 1):
            fine = coeffs[i]
            local_max = maximum_filter(coeffs[i + 1], size=coarse_neighborhood_size)
            local_min = minimum_filter(coeffs[i + 1], size=coarse_neighborhood_size)
            coarse_signed_max = np.where(
                np.abs(local_min) > np.abs(local_max),
                local_min,
                local_max,
            )
            tukey_window = _make_tukey_window(fine.shape)
            E = np.sum(np.abs(tukey_window * fine * coarse_signed_max))
            energies.append(max(E,0.0))
            metric +=max(E,0.0)

    elif product_mode == 'none':
        for i in range(coeffs.shape[0]):
            tukey_window = _make_tukey_window(coeffs[i].shape)
            E = np.sum(np.abs(coeffs[i]*tukey_window))
            energies.append(max(E,0.0))
            metric +=max(E,0.0)
    else:
        raise ValueError("Product mode must be one of ['absolute_values','signed_coefficients','none']")
    # energies = np.asarray(energies)
    # if product_mode == 'none':
    #     freqs = np.asarray(p['frequency_levels'])
    # else:
    #     freqs = np.asarray(p['frequency_levels'][:-1])
    # #metric = np.sum(energies * freqs) / np.sum(energies)
    return float(metric)

def make_isotropic_cwt2d_mexican_hat_bank(
    image_shape,
    pixel_size_um,
    NA,
    mode='widefield',
    wavelength_em=None,
    wavelength_ex=None,
    center_frequency_fractions = [0.75, 0.375, 0.1875, 0.09375, 0.046875] ,
    pad=True,
    pad_mode="reflect",
    pad_factor=6.0,
    normalization="unit_energy",
    dtype=np.float64,
):
    """
    Precompute isotropic Mexican-hat CWT filters using rFFT layout.
    This should be called once for a fixed:
        image shape,
        pixel size,
        cutoff frequency,
        frequency fractions,
        padding setting,
        normalization setting.
    """

    if len(image_shape) != 2:
        raise ValueError("image_shape must be a 2-tuple")

    image_ny, image_nx = image_shape

    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")

    cutoff_frequency = _compute_cutoff_frequency(mode=mode, NA=NA,
                                                wavelength_em=wavelength_em,
                                                wavelength_ex=wavelength_ex)
    if cutoff_frequency <= 0:
        raise ValueError("cutoff_frequency must be positive")

    fractions = np.asarray(center_frequency_fractions, dtype=dtype)

    if fractions.ndim != 1:
        raise ValueError("center_frequency_fractions must be a 1D array")

    if np.any(fractions <= 0):
        raise ValueError("all center_frequency_fractions must be positive")

    nyquist_frequency = 0.5 / pixel_size_um
    effective_cutoff = min(cutoff_frequency, nyquist_frequency)
    center_frequencies = fractions * effective_cutoff

    scales_um = np.sqrt(2.0) / (2.0 * np.pi * center_frequencies)
    scales_px = scales_um / pixel_size_um

    if pad:
        pad_px = int(np.ceil(pad_factor * np.max(scales_px)))
        pad_px = max(pad_px, 1)
    else:
        pad_px = 0

    padded_shape = (
        image_ny + 2 * pad_px,
        image_nx + 2 * pad_px,
    )

    ny, nx = padded_shape

    fy = np.fft.fftfreq(ny, d=pixel_size_um).astype(dtype)
    fx = np.fft.rfftfreq(nx, d=pixel_size_um).astype(dtype)

    FX, FY = np.meshgrid(fx, fy)
    radial_frequency = np.sqrt(FX**2 + FY**2)
    rho = 2.0 * np.pi * radial_frequency

    filters = np.empty(
        (len(center_frequencies), ny, nx // 2 + 1),
        dtype=dtype,
    )

    n_pixels = ny * nx
    rfft_weights = np.ones((ny, nx // 2 + 1), dtype=dtype)

    if nx % 2 == 0:
        if rfft_weights.shape[1] > 2:
            rfft_weights[:, 1:-1] = 2.0
    else:
        if rfft_weights.shape[1] > 1:
            rfft_weights[:, 1:] = 2.0

    for i, scale_um in enumerate(scales_um):
        u = scale_um * rho

        H = (u**2) * np.exp(-0.5 * u**2)
        H[0, 0] = 0.0

        if normalization == "unit_peak":
            H *= np.e / 2.0

        elif normalization == "cwt":
            H *= scale_um

        elif normalization == "unit_energy":
            spectral_energy = np.sum(rfft_weights * np.abs(H) ** 2)
            kernel_norm = np.sqrt(spectral_energy / n_pixels)

            if kernel_norm > 0:
                H /= kernel_norm

        elif normalization == "none":
            pass

        else:
            raise ValueError(
                "normalization must be one of "
                "{'unit_energy', 'unit_peak', 'cwt', 'none'}"
            )

        filters[i] = H

    half_power_low_ratio = 0.6169441746845508
    half_power_high_ratio = 1.4415132500606622

    bank = {
        "filters": filters,
        "image_shape": tuple(image_shape),
        "padded_shape": padded_shape,
        "pad": pad,
        "pad_px": pad_px,
        "pad_mode": pad_mode,
        "pad_factor": pad_factor,
        "pixel_size_um": pixel_size_um,
        "center_frequency_fractions": fractions,
        "center_frequencies": center_frequencies,
        "scales_um": scales_um,
        "scales_px": scales_px,
        "estimated_otf_cutoff": cutoff_frequency,
        "nyquist_frequency": nyquist_frequency,
        "effective_cutoff": effective_cutoff,
        "normalization": normalization,
        "half_power_low": half_power_low_ratio * center_frequencies,
        "half_power_high": half_power_high_ratio * center_frequencies,
        "dtype": dtype,
    }

    return bank

def _isotropic_cwt2d_mexican_hat_with_bank(
    image,
    bank,
    pad_mode=None,
    workers=-1,
):
    """
    Apply a precomputed isotropic Mexican-hat CWT bank to one image.
    """

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    if image.shape != bank["image_shape"]:
        raise ValueError(
            f"image shape {image.shape} does not match bank image_shape "
            f"{bank['image_shape']}"
        )

    pad_px = bank["pad_px"]

    if pad_mode is None:
        pad_mode = bank["pad_mode"]

    if pad_px > 0:
        work = np.pad(
            image,
            pad_width=((pad_px, pad_px), (pad_px, pad_px)),
            mode=pad_mode,
        )
    else:
        work = image

    image_fft = rfft2(work, workers=workers)

    filters = bank["filters"]

    coeffs_padded = np.empty(
        (filters.shape[0], work.shape[0], work.shape[1]),
        dtype=np.float64,
    )

    for i, H in enumerate(filters):
        coeffs_padded[i] = irfft2(
            image_fft * H,
            s=work.shape,
            workers=workers,
        )

    if pad_px > 0:
        coeffs = coeffs_padded[
            :,
            pad_px:-pad_px,
            pad_px:-pad_px,
        ]
    else:
        coeffs = coeffs_padded
    return coeffs

def _anscombe_transform(x):
    return 2.0 * np.sqrt(x + 3.0 / 8.0)

def _soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def _estimate_sigma_mad(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs = coeffs[np.isfinite(coeffs)]
    if coeffs.size == 0:
        return 0.0
    return np.median(np.abs(coeffs)) / 0.6745

def _compute_cutoff_frequency(mode, NA, wavelength_em, wavelength_ex = None):
    if mode == "widefield":
        return 2.0 * NA / wavelength_em

    elif mode == "confocal":
        if wavelength_ex is None:
            raise ValueError("Confocal requires excitation wavelength")
        return (2.0 * NA / wavelength_ex) + (2.0 * NA / wavelength_em)

    elif mode == "two_photon":
        if wavelength_ex is None:
            raise ValueError("Two-photon requires excitation wavelength")
        return 4.0 * NA / wavelength_ex

    else:
        raise ValueError(f"Unknown mode: {mode}")

