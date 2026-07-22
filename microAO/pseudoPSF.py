import numpy as np
from numpy.fft import *
from copy import copy
from skimage import restoration
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
import pywt
import math as m
from scipy.signal.windows import tukey


def make_pairs(n:int):
    pairs = []
    for i in range(0,n,2):
        pairs.append((i,i+1))
        pairs.append((i+1,i))
    return pairs



def _anscombe_transform(x):
  
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, 0.0)
    return 2.0 * np.sqrt(x + 3.0 / 8.0)


def _extract_swt_detail_stack(imagestack,wavelet="sym2",levels=3,):

    coeffs = pywt.swt2(
        imagestack,
        wavelet,
        level=levels,
        axes=(0, 1),
        norm=True,
        trim_approx=True,
    )
    detail_levels = coeffs[1:][::-1]  # fine -> coarse

    details = np.stack(
        [np.stack(detail_tuple, axis=0) for detail_tuple in detail_levels],
        axis=0,
    )
    # shape: (levels, 3, H, W, C)
    details = np.moveaxis(details, -1, 0)
    # shape: (C, levels, 3, H, W)
    return details.astype(np.float32, copy=False)


def _subband_wiener_shrink(coeffs, eps=1e-12):

    coeffs = coeffs.astype(np.float32, copy=False)

    med = np.median(coeffs, axis=(-2, -1), keepdims=True)
    sigma = np.median(
        np.abs(coeffs - med),
        axis=(-2, -1),
        keepdims=True,
    ) / 0.67448975
    noise_var = sigma**2
    total_var = np.mean(coeffs**2, axis=(-2, -1), keepdims=True)
    signal_var = np.maximum(total_var - noise_var, 0.0)

    gain = signal_var / (signal_var + noise_var + eps)

    return coeffs * gain


def _stabilized_subband_ratio(numerator,denominator, reg=0.05,eps=1e-12,center=True):
    """
    Computes:
        F(num) * conj(F(den)) / (abs(F(den))**2 + floor)
    """
    numerator = numerator.astype(np.float32, copy=False)
    denominator = denominator.astype(np.float32, copy=False)

    H, W = numerator.shape[-2:]

    F0 = np.fft.rfft2(numerator, axes=(-2, -1))
    F1 = np.fft.rfft2(denominator, axes=(-2, -1))

    power = np.abs(F1) ** 2

    # One relative floor per pair / level / orientation plane.
    floor = reg * np.median(power, axis=(-2, -1), keepdims=True)

    ratio = F0 * np.conj(F1) / (power + floor + eps)

    out = np.fft.irfft2(
        ratio,
        s=(H, W),
        axes=(-2, -1),
    ).real

    if center:
        out = np.fft.ifftshift(out, axes=(-2, -1))

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out.astype(np.float32, copy=False)




def make_deconvolution_pairs(N):
    if N % 2 != 0:
        raise ValueError("N must be even")
    pairs = []
    for i in range(0, N-1, 2):
        pairs.extend([(i,   i+1),
                      (i+1, i  )])
    return pairs

def neighshrink(coefs, lam, base_kernel_size=3):
 
    N, L, O, H, W = coefs.shape
    sq = coefs**2                       

   
    sum_sqs = np.empty_like(sq)

    for l in range(L):
        k = base_kernel_size**(l + 1)
        local_mean = uniform_filter(
            sq[:, l, :, :, :],
            size=(1, 1, k, k),
            mode='reflect'
        )
        sum_sqs[:, l, :, :, :] = local_mean * (k * k)

    lam2 = lam**2
    lam2 = lam2[:, None, None, None, None]  

    sum_sqs = np.maximum(sum_sqs, 0) #There is some floating point errors going on here because I get very small negative numbers for some reason
    with np.errstate(divide='ignore', invalid='ignore'):
        shrink = 1.0 - lam2 / sum_sqs
        shrink[sum_sqs == 0] = 0

    shrink = np.maximum(shrink, 0)

    return shrink * coefs


def swt_denoise(image,levels=3):

    def _neighshrink2(arr, kernel_size, lam):
        squared = arr**2
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
        sum_of_squares = convolve2d(squared, kernel, mode='same', boundary='fill', fillvalue=0)

        sum_of_squares = np.maximum(sum_of_squares, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            shrink = 1.0 - lam**2 / sum_of_squares
            shrink[sum_of_squares == 0] = 0
        
        shrink = np.maximum(shrink, 0)


        return shrink * arr



    image=2*np.sqrt(image+(3/8))
    sigma=restoration.estimate_sigma(image)
    lam=m.sqrt(2*sigma**2*m.log10(len(image)**2))
    coeffs=pywt.swt2(image, 'sym2',level=levels,norm=True)
    denoise_coeffs=[]

    j=levels-1
    for Level in coeffs:
        denoised_level=[]
        CB,CD=Level
        # CB=_neighshrink2(CB, 3*(2**j), lam)
        CB=np.zeros_like(CB)
        for C in CD:
           
            C=_neighshrink2(C, 3*(2**j), lam)
            denoised_level.append(C)

        denoise_coeffs.append((CB,tuple(denoised_level)))
        j=j-1

    denoised_image=pywt.iswt2(denoise_coeffs,'sym2',norm=True)
    denoised_image=((denoised_image/2)**2-(3/8))
    return denoised_image




def pseudoPSF(batch, pairs, pseudopsfsize: int = 32, mode='default',reg_parameter=0.1):
    pseudoPSFbatch=np.zeros((batch.shape[0],int(pseudopsfsize),int(pseudopsfsize),len(pairs)))
    pseudopsf = np.zeros((int(pseudopsfsize),int(pseudopsfsize),len(pairs)))
    
    imagestackshape =batch[0].shape
    assert imagestackshape[0] == imagestackshape[1]
    uppx = round(imagestackshape[0]/2-pseudopsfsize/2)
    downpx = uppx+pseudopsfsize
    leftpx = round(imagestackshape[1]/2-pseudopsfsize/2)
    rightpx = leftpx+pseudopsfsize
    
    
    for n, imagestack in enumerate(batch):
        

        if mode == 'default':
            fourierstack = rfft2(copy(imagestack),axes=(0,1))
            for i, index in enumerate(pairs):
                div = np.divide(fourierstack[:,:,index[0]], fourierstack[:,:,index[1]])
                im = ifftshift(irfft2(div)).real
                pseudopsf[:,:,i] = im[uppx:downpx, leftpx:rightpx]
            pseudoPSFbatch[n,:,:,:]=pseudopsf
            continue

        elif mode == 'wiener':
            anscombe=2*np.sqrt(imagestack+(3/8))
            for i, index in enumerate(pairs):
                im = restoration.wiener(anscombe[:,:,index[0]], anscombe[:,:,index[1]], reg_parameter)
                pseudopsf[:,:,i] = im[uppx:downpx, leftpx:rightpx]
            pseudoPSFbatch[n,:,:,:]=pseudopsf
            continue
        
        elif mode == 'wavelet':
            levels = 3
            wavelet_name = "sym2"

            H, W, C = imagestack.shape

            if H != W:
                raise ValueError("Wavelet pseudoPSF branch expects square images.")

            if H % (2**levels) != 0 or W % (2**levels) != 0:
                raise ValueError(
                    f"SWT level={levels} requires image dimensions divisible by "
                    f"{2**levels}. Got image shape {(H, W)}."
                )

            pair_array = np.asarray(pairs, dtype=np.int64)
            pair_num = pair_array[:, 0]
            pair_den = pair_array[:, 1]

            anscombe = _anscombe_transform(imagestack)
            coeffs = _extract_swt_detail_stack(
                anscombe,
                wavelet=wavelet_name,
                levels=levels,
            )

            coeffs = _subband_wiener_shrink(coeffs)
            numerator = coeffs[pair_num]
            denominator = coeffs[pair_den]
            deconv = _stabilized_subband_ratio(
                numerator,
                denominator,
                reg=reg_parameter,
                eps=1e-12,
                center=True,
            )
            level_weights = 1.0 / (4.0 ** np.arange(levels, dtype=np.float32))
            deconv *= level_weights[None, :, None, None, None]
            pseudo_full = np.sum(deconv, axis=(1, 2))
            crop = pseudo_full[:, uppx:downpx, leftpx:rightpx]
            pseudopsf[:, :, :] = np.moveaxis(crop, 0, -1)
            pseudoPSFbatch[n, :, :, :] = pseudopsf
            continue


        elif mode == 'SWTdenoise':
            denoisedstack=[]
            imagestack = np.moveaxis(imagestack, 2, 0)
            for image in imagestack:
                denoisedstack.append(swt_denoise(image))
            denoisedstack = np.stack(denoisedstack, axis=2)
            for i, index in enumerate(pairs):
                im = restoration.wiener(denoisedstack[:,:,index[0]], denoisedstack[:,:,index[1]], reg_parameter)
                pseudopsf[:,:,i] = im[uppx:downpx, leftpx:rightpx]
            pseudoPSFbatch[n,:,:,:]=pseudopsf
            continue

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be one of ['default', 'wiener', 'wavelet','SWTdenoise'].")
 

    return pseudoPSFbatch.astype('float32')

def tukey_window(image, feather=0.1):

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
        _, rows, cols = img.shape
        wy = tukey(rows, feather, sym=True)
        wx = tukey(cols, feather, sym=True)
        window2d = np.outer(wy, wx)
        return img * window2d[np.newaxis, :, :]