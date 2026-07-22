#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2018 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>
##
## microAO is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## microAO is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with microAO.  If not, see <http://www.gnu.org/licenses/>.

#Import required packs
import dataclasses
import numpy as np
from scipy.signal import tukey
from skimage.filters import threshold_otsu
from microAO.aoWaveletMetric import wavelet_image_quality_metric
from microAO.oldAoWaveletMetric import old_wavelet_image_quality_metric


@dataclasses.dataclass(frozen=True)
class DiagnosticsFourier:
    fft_sq_log: np.ndarray
    freq_above_noise: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsContrast:
    image_raw: np.ndarray
    mean_top: float
    mean_bottom: float

@dataclasses.dataclass(frozen=True)
class DiagnosticsGradient:
    image_raw: np.ndarray
    grad_mask_x: np.ndarray
    grad_mask_y: np.ndarray
    correction_grad: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsFourierPower:
    fftarray_sq_log: np.ndarray
    freq_above_noise: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsSecondMoment:
    fftarray_sq_log: np.ndarray
    fftarray_sq_log_masked: np.ndarray

@dataclasses.dataclass(frozen=True)
class DiagnosticsWavelet:
    placeholder: np.ndarray

def _tukey_window(image, feather=0.1):

        img = np.asarray(image)

        if img.ndim == 2:
            # Single image
            rows, cols = img.shape
            wy = tukey(rows, feather, sym=True)
            wx = tukey(cols, feather, sym=True)
            window2d = np.outer(wy, wx)
            return img * window2d

        elif img.ndim == 3:
            # Stack of images
            n_slices, rows, cols = img.shape
            wy = tukey(rows, feather, sym=True)
            wx = tukey(cols, feather, sym=True)
            window2d = np.outer(wy, wx)
            return img * window2d[np.newaxis, :, :]





def make_OTF_mask(size, inner_rad, outer_rad):
    rad_y = int(size[0] / 2)
    rad_x = int(size[1] / 2)

    outer_mask = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                         np.arange(-rad_x, rad_x) ** 2) < outer_rad

    inner_mask_neg = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                         np.arange(-rad_x, rad_x) ** 2) < inner_rad
    inner_mask = (inner_mask_neg - 1) * -1
    ring_mask = outer_mask * inner_mask
    return ring_mask

def find_noise_level(image, wavelength, NA, pixel_size, noise_amp_factor=1.125):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = (freq_ratio) * (np.max(image.shape) / 2)
    im_shift = np.fft.fftshift(image)

    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = make_OTF_mask(np.shape(image), 0, 1.1 * OTF_outer_rad)
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * noise_amp_factor
    return threshold

def measure_fourier_metric(image, wavelength, NA, pixel_size, fourier_noise_level, **kwargs):

   #Pad and feather image to a square
    h,w=np.shape(image)
    patch = _tukey_window(image)
    desired_side = max(w, h) 
    x_insert=(desired_side-w)//2
    y_insert=(desired_side-h)//2
    image=np.full((desired_side, desired_side), 0, dtype=patch.dtype)
    image[y_insert:y_insert + h, x_insert:x_insert + w] = patch

    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = (freq_ratio) * (np.max(image.shape) / 2)
    
    fftarray = np.fft.fftshift(np.fft.fft2(image)) #Might need to remove the shift here
    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

   
    OTF_mask = make_OTF_mask(np.shape(image), 0.1 * OTF_outer_rad, OTF_outer_rad)
    freq_above_noise = (fftarray_sq_log > fourier_noise_level) * OTF_mask
    metric = np.count_nonzero(freq_above_noise)
    return metric, DiagnosticsFourier(fftarray_sq_log, freq_above_noise)

def measure_contrast_metric(image, percent=1.0, **kwargs):
    img = np.asarray(image, dtype=np.float32)
    p = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
    smoothed = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:  ] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:  ] +
        p[2:  , 0:-2] + p[2:  , 1:-1] + p[2:  , 2:  ]
    ) / 9.0
    frac = percent / 100.0 if percent >= 1.0 else percent
    if not (0.0 < frac < 0.5):
        raise ValueError("percent must correspond to a fraction in (0, 0.5).")
    flat = smoothed.ravel()
    n = flat.size
    k = max(1, int(np.ceil(frac * n)))
    sorted_flat = np.sort(flat)
    sum_bottom = float(np.sum(sorted_flat[:k]))
    sum_top = float(np.sum(sorted_flat[-k:]))
    eps = np.finfo(np.float32).eps
    denom = sum_bottom if abs(sum_bottom) > eps else eps
    ratio = sum_top / denom
    return (
        ratio,
        DiagnosticsContrast(
            image,   
            sum_top,
            sum_bottom
        )
    )

def measure_gradient_metric(image, **kwargs):
    image_gradient_x = np.gradient(image, axis=1)
    image_gradient_y = np.gradient(image, axis=0)

    grad_mask_x = image_gradient_x > (threshold_otsu(image_gradient_x) * 1.125)
    grad_mask_y = image_gradient_y > (threshold_otsu(image_gradient_y) * 1.125)

    correction_grad = np.sqrt((image_gradient_x * grad_mask_x) ** 2 + (image_gradient_y * grad_mask_y) ** 2)

    metric = np.mean(correction_grad)
    return metric, DiagnosticsGradient(image, grad_mask_x, grad_mask_y, correction_grad)


def measure_fourier_power_metric(image, wavelength, NA, pixel_size, noise_amp_factor=1.125,
                                 high_f_amp_factor=100, **kwargs):
    

    #Pad and feather image to a square
    h,w=np.shape(image)
    patch = _tukey_window(image)
    desired_side = max(w, h) 
    x_insert=(desired_side-w)//2
    y_insert=(desired_side-h)//2
    image=np.full((desired_side, desired_side), 0, dtype=patch.dtype)
    image[y_insert:y_insert + h, x_insert:x_insert + w] = patch




    
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = freq_ratio * (np.max(np.shape(image)) / 2)


    fftarray = np.fft.fftshift(np.fft.fft2(image))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = make_OTF_mask(np.shape(image), 0, 1.1 * OTF_outer_rad)
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * noise_amp_factor

    circ_mask = make_OTF_mask(np.shape(image), 0, OTF_outer_rad)

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    x_p = x - ((image.shape[1] - 1) / 2)
    x_prime = np.outer(np.ones(image.shape[0]), x_p)
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y_p = y - ((image.shape[0] - 1) / 2)
    y_prime = np.outer(y_p, np.ones(image.shape[1]))
    ramp_mask = x_prime ** 2 + y_prime ** 2

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    dist = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                   np.arange(-rad_x, rad_x) ** 2)
    omega = 1 - np.exp((dist / OTF_outer_rad) - 1)

    high_f_amp_mask = high_f_amp_factor * (ramp_mask * omega) / np.max(ramp_mask * omega)

    OTF_mask = make_OTF_mask(np.shape(image), 0.1 * OTF_outer_rad, OTF_outer_rad)
    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask * high_f_amp_mask
    metric = np.sum(freq_above_noise)
    return metric, DiagnosticsFourierPower(fftarray_sq_log, freq_above_noise)


def measure_second_moment_metric(image, wavelength, NA, pixel_size, **kwargs):

    #Pad and feather image to a square
    h,w=np.shape(image)
    patch = _tukey_window(image)
    desired_side = max(w, h) 
    x_insert=(desired_side-w)//2
    y_insert=(desired_side-h)//2
    image=np.full((desired_side, desired_side), 0, dtype=patch.dtype)
    image[y_insert:y_insert + h, x_insert:x_insert + w] = patch



    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = freq_ratio * (np.max(np.shape(image)) / 2)

  
    fftarray = np.fft.fftshift(np.fft.fft2(image))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    ring_mask = make_OTF_mask(np.shape(image), 0, OTF_outer_rad)

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    x_p = x - ((image.shape[1] - 1) / 2)
    x_prime = np.outer(np.ones(image.shape[0]), x_p)
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y_p = y - ((image.shape[0] - 1) / 2)
    y_prime = np.outer(y_p, np.ones(image.shape[1]))
    ramp_mask = x_prime ** 2 + y_prime ** 2

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    dist = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                   np.arange(-rad_x, rad_x) ** 2)
    omega = 1 - np.exp((dist/OTF_outer_rad)-1)

    fftarray_sq_log_masked = ring_mask * fftarray_sq_log * ramp_mask * omega
    metric = np.sum(fftarray_sq_log_masked)/np.sum(fftarray_sq_log)
    return metric, DiagnosticsSecondMoment(fftarray_sq_log, fftarray_sq_log_masked)

#ANDREI's wavelet metric (see aoWaveletMetric for actually implementation)
def measure_wavelet_metric(
    image,
    wavelength,
    NA,
    pixel_size,
    wavelength_ex=None,
    mode="widefield",
    params=None,
    **kwargs
):
    filter_bank_parameters = {'NA':NA,'wavelength_em':wavelength*(10**6),'wavelength_ex':wavelength_ex,
                              'pixel_size_um':pixel_size*(10**6),'mode':mode}
    return  wavelet_image_quality_metric(image,filter_bank_params=filter_bank_parameters), DiagnosticsWavelet(np.zeros((50,50)))

def measure_old_wavelet_metric(
    image,
    wavelength,
    NA,
    pixel_size,
    wavelength_ex=None,
    mode="widefield",
    params=None,
    **kwargs
):
   
    return old_wavelet_image_quality_metric(_tukey_window(image),NA,pixel_size*(10**6),wavelength*(10**6),wavelength*(10**6)), DiagnosticsWavelet(np.zeros((50,50)))


metric_function = {
    'fourier': measure_fourier_metric,
    'contrast': measure_contrast_metric,
    'fourier_power': measure_fourier_power_metric,
    'gradient': measure_gradient_metric,
    'second_moment': measure_second_moment_metric,
    'wavelet' : measure_wavelet_metric,
    'old_wavelet' : measure_old_wavelet_metric,
}

metric_names = dict([
    ('fourier', "Fourier Metric",),
    ('contrast', "Contrast metric"),
    ('fourier_power', "Fourier Power metric"),
    ('gradient', "Gradient metric"),
    ('second_moment', "Second Moment metric"),
    ('wavelet','Wavelet metric'),
    ('old_wavelet','Old wavelet metric')
])
