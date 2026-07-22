import numpy as np
import pywt
from scipy.ndimage import maximum_filter



#_____________HELPERS_______________

def anscombe_transform(x):
    return 2.0 * np.sqrt(x + 3.0 / 8.0)

def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def estimate_sigma_mad(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeffs = coeffs[np.isfinite(coeffs)]
    if coeffs.size == 0:
        return 0.0
    return np.median(np.abs(coeffs)) / 0.6745

def universal_threshold(sigma, n_coeffs):
    if n_coeffs <= 1 or sigma <= 0:
        return 0.0
    return sigma * np.sqrt(2.0 * np.log(n_coeffs))

def swt_nominal_band(level, pixel_size_um):
    fs = 1.0 / pixel_size_um
    f_high = fs / (2 ** level)
    f_low = fs / (2 ** (level + 1))
    f_center = 0.5 * (f_low + f_high)
    return f_low, f_center, f_high

def max_swt_level(shape):
    def pow2(n):
        k = 0
        while n % 2 == 0 and n > 0:
            n //= 2
            k += 1
        return k
    return min(pow2(shape[0]), pow2(shape[1]))

def compute_cutoff_frequency(mode, NA, wavelength_em, wavelength_ex):
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

def pad_to_power_of_two_square(image: np.ndarray,) -> np.ndarray:

    height, width = image.shape
    largest_dimension = max(height, width)

    target_size = 1 << (largest_dimension - 1).bit_length()

    height_padding = target_size - height
    width_padding = target_size - width

    pad_top = height_padding // 2
    pad_bottom = height_padding - pad_top
    pad_left = width_padding // 2
    pad_right = width_padding - pad_left

    return np.pad(
        image,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=np.mean(image),
    )
#___________________MAIN METRIC__________________

def old_wavelet_image_quality_metric(
    image,
    NA,
    pixel_size_um,
    wavelength_em,
    wavelength_ex,
    mode="widefield",
    params=None,
):
    """
    Computes an image quality metric based on the persistance of wavelet coefficients across length scales.
    The metric is calculated by computing persistance coefficients across all adjancent scale levels, then taking the sum
    for each scale level. The centroid in frequency space of the the persistance coefficients is computed, and the metric
    is defined as the position of this centroid, with positions tending towards higher frequency bands being ranked higher.
    Denoising is also directly incorporated before the persistance calculation, to further help robustness.

    Input:
    image -- 2D array of a grayscale image
    NA -- microscope NA, used in OTF estimation
    pixel_size_um -- size of pixels in the image in the sample plane, measured in micrometers, used in OTF estimation
    wavelength_em -- emission wavelenght, ignored if the imaging mode is set to two_photon, used in OTF estimation
    wavelength_ex -- excition wavelength, ignored if the imaging mode is set to widefield, used in OTF estimation
    mode -- can be one of 'widefield' 'confocal' or 'two_photon', changes OTF computation
    params -- dict allowing for the modification of the default metric parameters; includes:
        wavelet -- wavelet to use in decomposition
        n_levels -- number of levels in the wavelet decomposition, note that if the image is too small, the 
                real number of levels might be lower
        coarse_neighborhood -- when computing persistency, the coarse detail level gets filtered using a square max filter,
                               this parameter determines the size of this filter
        denoise_mode -- how denoising should be carried out on the wavelet coefficients, can be 'off' or 'anscombe_soft'
        denoise_strength -- how strong the denoising effect should be, if enbabled, from 0.0 to 1.0
        
       
    Output:
    metric - image quality metric

    """

    defaults = dict(
        wavelet="db2",
        n_levels=5,
        coarse_neighborhood=3,
        denoise_mode="off",         
        denoise_strength=0.8,
    )

    if params is not None:
        defaults.update(params)

    p = defaults

    work_image = pad_to_power_of_two_square(image)

    cutoff = compute_cutoff_frequency(
        mode=mode,
        NA=NA,
        wavelength_em=wavelength_em,
        wavelength_ex=wavelength_ex,
    )


    max_levels = max_swt_level(work_image.shape)
    n_levels = min(p["n_levels"], max_levels)
    print('max_levels,n_levels')
    print(max_levels)
    print(n_levels)
    if p["denoise_mode"] == "anscombe_soft":
        work_image = anscombe_transform(work_image)
   

    coeffs = pywt.swt2(work_image, wavelet=p["wavelet"], level=n_levels)

    coeffs_by_level = {
        n_levels - i: c for i, c in enumerate(coeffs)
    }

    
    level_info = []
    for level in range(1, n_levels + 1):
        _, f_center, _ = swt_nominal_band(level, pixel_size_um)
        keep = f_center <= cutoff
        level_info.append((level, f_center, keep))


    sigma = 0.0
    threshold = 0.0

    if p["denoise_mode"] == "anscombe_soft":
        invalid = []

        for (level, _, keep) in level_info:
            if not keep:
                _, (cH, cV, cD) = coeffs_by_level[level]
                invalid.extend([cH.ravel(), cV.ravel(), cD.ravel()])

        if len(invalid) > 0:
            sigma = estimate_sigma_mad(np.concatenate(invalid))
        else:
            _, (cH, cV, cD) = coeffs_by_level[1]
            sigma = estimate_sigma_mad(
                np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
            )

        threshold = p["denoise_strength"] * universal_threshold(
            sigma, image.size
        )

   
    detail_coeffs = {}
    for level in range(1, n_levels + 1):
        _, detail = coeffs_by_level[level]

        if threshold > 0:
            detail = tuple(soft_threshold(c, threshold) for c in detail)

        detail_coeffs[level] = detail


    energies = []
    freqs = []
    npx = work_image.shape[0]*work_image.shape[1]
    for i in range(len(level_info) - 1):
        l1, f1, keep1 = level_info[i]
        l2, _, keep2 = level_info[i + 1]

        if not (keep1 and keep2):
            continue
        E=0
        for j in range(3):
            fine = np.abs(detail_coeffs[l1][j])
            coarse = np.abs(detail_coeffs[l2][j])
            coarse_max = maximum_filter(coarse, size=p["coarse_neighborhood"])
            persistence = fine * coarse_max
            E += np.sum(persistence) - npx*np.mean(fine)*np.mean(coarse_max)
        energies.append(E)
        freqs.append(f1)
    
    if len(energies) == 0:
        return 0.0

    energies = np.asarray(energies)
    freqs = np.asarray(freqs)

    metric = np.sum(energies * freqs) / np.sum(energies)
    return float(metric)