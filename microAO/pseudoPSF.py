import numpy as np
from numpy.fft import *
from copy import copy
from skimage import restoration
from scipy.ndimage import uniform_filter
import pywt

import functools, time
def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__}: {(t1 - t0)*1000:.2f} ms")
        return result
    return wrapper

def make_pairs(N):
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

def pseudoPSF(batch, biasmode, pairs, pseudopsfsize: int = 32, mode='default'):
    pseudoPSFbatch=np.zeros((batch.shape[0],int(pseudopsfsize),int(pseudopsfsize),len(pairs)))
    pseudopsf = np.zeros((int(pseudopsfsize),int(pseudopsfsize),len(pairs)))
    
    imagestackshape =batch[0].shape
    assert imagestackshape[-1]>= len(biasmode)
    assert imagestackshape[0] == imagestackshape[1]
    uppx = round(imagestackshape[0]/2-16)
    downpx = uppx+32
    leftpx = round(imagestackshape[1]/2-16)
    rightpx = leftpx+32
    
    
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
                im = restoration.wiener(anscombe[:,:,index[0]], anscombe[:,:,index[1]], 0.5)
                pseudopsf[:,:,i] = im[uppx:downpx, leftpx:rightpx]
            pseudoPSFbatch[n,:,:,:]=pseudopsf
            continue

        elif mode == 'wavelet':
            
            anscombe=2*np.sqrt(imagestack+(3/8)) #Normalize noise
            sigma=np.array(restoration.estimate_sigma(anscombe,channel_axis=-1))
            lam=np.sqrt(2*sigma**2*np.log10(imagestackshape[0]**2)) #Threshold for denoising using NeighShrink
            coeffs=pywt.swt2(anscombe, 'sym2',level=3, axes=(0,1),norm=True,trim_approx=True)
            coeffs.pop(0) #Remove Background

            coeffs=np.array(list(reversed(coeffs))).transpose(4,0,1,2,3) #Shape of (num_images,levels,HVD,pixels_x,pixels_y)
            coeffs_shape=coeffs.shape
            coeffs=neighshrink(coeffs,lam)
            for i, index in enumerate(pairs):
                coeffs0=coeffs[index[0]]
                coeffs1=coeffs[index[1]]
                deconv_coeffs=np.empty_like(coeffs0)
                for l in range(coeffs_shape[1]):
                    for o in range(coeffs_shape[2]):
                        deconv_coeffs[l,o,:,:]=restoration.wiener(coeffs0[l,o,:,:],coeffs1[l,o,:,:],0.5)
                   
                im=np.sum(deconv_coeffs,axis=(0,1))
                pseudopsf[:,:,i] = im[uppx:downpx, leftpx:rightpx]

            pseudoPSFbatch[n,:,:,:]=pseudopsf
            continue


        else:
            raise ValueError(f"Unknown mode: {mode}. Must be one of ['default', 'wiener', 'wavelet'].")
 

    return pseudoPSFbatch.astype('float32')
