#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2018 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>, Josh Edwards
## <Josh.Edwards222@gmail.com> & Jacopo Antonello
## <jacopo.antonello@dpag.ox.ac.uk>
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
from abc import abstractmethod
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import tukey
from scipy.optimize import curve_fit

from skimage.filters import threshold_otsu
import aotools
import scipy.stats as stats
from skimage.restoration import unwrap_phase
from scipy.integrate import trapz
from microAO.aoMetrics import metric_function
from microAO.events import *


class AdaptiveOpticsFunctions():

    def __init__(self):
        self.mask = None
        self.fft_filter = None
        self.controlMatrix = None
        self.metric = 'fourier'

    def set_mask(self,mask):
        self.mask = mask
        return

    def set_fft_filter(self,fft_filter):
        self.fft_filter = fft_filter
        return

    def set_controlMatrix(self, controlMatrix):
        self.controlMatrix = controlMatrix
        return

    def make_mask(self, radius):
        diameter = radius * 2
        self.mask = np.sqrt((np.arange(-radius,radius)**2).reshape((diameter,1)) + (np.arange(-radius,radius)**2)) < radius
        return self.mask


    def bin_ndarray(self, ndarray, new_shape, operation='mean'):
        """

        Function acquired from Stack Overflow: https://stackoverflow.com/a/29042041. Stack Overflow or other Stack Exchange
        sites is cc-wiki (aka cc-by-sa) licensed and requires attribution.

        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.

        Number of output dimensions must match number of input dimensions and
            new axes must divide old ones.

        Example
        -------

        m = np.arange(0,100,1).reshape((10,10))
        n = bin_ndarray(m, new_shape=(5,5), operation='sum')
        print(n)

        [[ 22  30  38  46  54]
         [102 110 118 126 134]
         [182 190 198 206 214]
         [262 270 278 286 294]
         [342 350 358 366 374]]

        """
        operation = operation.lower()
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                               new_shape))
        compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                      ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))
        return ndarray

    def mgcentroid(self, myim, mythr=0.0):
        assert(myim.dtype == np.float)

        myn1, myn2 = myim.shape
        myxx1, myxx2 = np.meshgrid(range(1, myn1 + 1), range(1, myn2 + 1))
        myim[myim < mythr] = 0
        mysum1 = np.sum((myxx1*myim).ravel())
        mysum2 = np.sum((myxx2*myim).ravel())
        mymass = np.sum(myim.ravel())
        return int(np.round(mysum1/mymass)), int(np.round(mysum2/mymass))

    def make_fft_filter(self, image, region=None, window_dim = None, mask_di = None):
        # Convert image to array and float
        data = np.asarray(image)

        if region is None:
            region = int(data.shape[0]/16)

        # Apply tukey window
        fringes = np.fft.fftshift(data)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        # Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        # Remove center section to allow finding of 1st order point
        fftarray = np.fft.fftshift(fftarray)
        find_cent = [int(fftarray.shape[1]/2),int(fftarray.shape[0]/ 2)]
        fftarray[find_cent[1]-region:find_cent[1]+region,find_cent[0]-region:find_cent[0]+region]=0.00001+0j

        # Find approximate position of first order point
        test_point = np.argmax(fftarray[:fftarray.shape[0] // 2, :])
        test_point = [int(test_point % fftarray.shape[1]), int(test_point / fftarray.shape[1])]

        # Find first order point
        maxpoint = np.zeros(np.shape(test_point), dtype=int)
        maxpoint[:] = test_point[:]
        if window_dim is None:
            window_dim = 100
        elif window_dim % 2 != 0:
            window_dim += 1
        else:
            pass
        window = np.zeros((window_dim, window_dim))

        for ii in range(10):
            try:
                window[:, :] = np.log(
                    abs(fftarray[maxpoint[1] - window_dim // 2:maxpoint[1] + window_dim // 2,
                        maxpoint[0] - window_dim // 2:maxpoint[0] + window_dim // 2]))
            except ValueError as e:
                raise Exception(
                    "Interferometer stripes are too fine. Please make them coarser or reduce window size").with_traceback(e.__traceback__)
            thresh = threshold_otsu(window)
            binaryIm = window > thresh
            windowOtsu = window * binaryIm
            CoM = np.zeros((1, 2))
            CoM[0, :] = np.round(center_of_mass(windowOtsu))
            maxpoint[0] = maxpoint[0] - window_dim // 2 + int(CoM[0, 1])
            maxpoint[1] = maxpoint[1] - window_dim // 2 + int(CoM[0, 0])

        self.fft_filter = np.zeros(np.shape(fftarray))
        if mask_di is None:
            mask_di = int(data.shape[0] * (5.0 / 16.0))
        elif type(mask_di) is not int:
            mask_di = int(mask_di)
        else:
            pass

        x_shift = np.min((0, abs(maxpoint[0] - mask_di), abs(maxpoint[0] - fftarray.shape[0]) - mask_di))
        y_shift = np.min((0, abs(maxpoint[1] - mask_di), abs(maxpoint[1] - fftarray.shape[1]) - mask_di))

        x = np.sin(np.linspace(0, np.pi, mask_di)) ** 2
        fourier_mask = np.outer(x, x.T)
        y_min = maxpoint[1] - int(np.floor((mask_di / 2.0))) + y_shift
        y_max = maxpoint[1] + int(np.ceil((mask_di / 2.0))) + y_shift
        x_min = maxpoint[0] - int(np.floor((mask_di / 2.0))) + x_shift
        x_max = maxpoint[0] + int(np.ceil((mask_di / 2.0))) + x_shift

        self.fft_filter[y_min:y_max, x_min:x_max] = fourier_mask
        self.fft_filter = np.roll(self.fft_filter, -x_shift, axis=1)
        self.fft_filter = np.roll(self.fft_filter, -y_shift, axis=0)
        return self.fft_filter

    def unwrap_interferometry(self, image, **kwargs):
        #Convert image to array and float
        data = np.asarray(image)

        #Apply tukey window
        fringes = np.fft.fftshift(data)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1)*tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        #Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        #Apply Fourier filter
        M = np.fft.fftshift(self.fft_filter)
        fftarray_filt = fftarray * M
        fftarray_filt = np.fft.fftshift(fftarray_filt)

        #Roll data to the centre
        centre_y_array, centre_x_array = np.where(self.fft_filter == np.max(self.fft_filter))
        g1 = int(np.round(np.mean(centre_y_array)) - np.round(fftarray_filt.shape[0] // 2))
        g0 = int(np.round(np.mean(centre_x_array)) - np.round(fftarray_filt.shape[0] // 2))
        fftarray_filt = np.roll(fftarray_filt, -g0, axis=1)
        fftarray_filt = np.roll(fftarray_filt, -g1, axis=0)

        #Convert to real space
        fftarray_filt_shift = np.fft.fftshift(fftarray_filt)
        complex_phase = np.fft.fftshift(np.fft.ifft2(fftarray_filt_shift))

        #Find phase data by taking 2d arctan of imaginary and real parts
        phaseorder1 = np.zeros(complex_phase.shape)
        phaseorder1[:,:] = np.arctan2(complex_phase.imag,complex_phase.real)

        #Mask out edge region to allow unwrap to only use correct region
        phaseorder1mask = phaseorder1 * self.mask

        #Perform unwrap
        phaseorder1unwrap = unwrap_phase(phaseorder1mask)
        phase_image = phaseorder1unwrap * self.mask
        return phase_image

    def get_zernike_modes(self, image_unwrap, noZernikeModes, resize_dim = 128):
        #Resize image
        original_dim = int(np.shape(image_unwrap)[0])
        while original_dim%resize_dim != 0:
            resize_dim -= 1

        if resize_dim < original_dim/resize_dim:
            resize_dim = int(np.round(original_dim/resize_dim))

        image_resize = self.bin_ndarray(image_unwrap, new_shape=(resize_dim,resize_dim), operation='mean')

        #Calculate Zernike mode
        zcoeffs_dbl = []
        num_pixels = np.count_nonzero(aotools.zernike.zernike_noll(1, resize_dim))
        for i in range(1,(noZernikeModes+1)):
            intermediate = trapz(image_resize * aotools.zernike.zernike_noll(i, resize_dim))
            zcoeffs_dbl.append(trapz(intermediate) / (num_pixels))
        coef = np.asarray(zcoeffs_dbl)
        return coef

    def create_control_matrix(self, zernikeAmps, pokeSteps, numActuators, pupil_ac=None, threshold=0.005):
        if np.any(pupil_ac) == None:
            pupil_ac = np.ones(numActuators)

        noZernikeModes = zernikeAmps.shape[1]

        slopes = np.zeros(noZernikeModes)
        intercepts = np.zeros(noZernikeModes)
        r_values = np.zeros(noZernikeModes)
        p_values = np.zeros(noZernikeModes)
        std_errs = np.zeros(noZernikeModes)

        C_mat = np.zeros((noZernikeModes, numActuators))
        offsets = np.zeros((noZernikeModes, numActuators))
        P_tests = np.zeros((noZernikeModes, numActuators))

        for ii in range(numActuators):
            if pupil_ac[ii] == 1:
                pokeSteps_trimmed = pokeSteps[np.where(pokeSteps[:, ii] != 0)[0], ii]
                zernikeModeAmp = zernikeAmps[np.where(pokeSteps[:, ii] != 0)[0], :]

                # Check that the influence slope for each actuator can actually be calculated
                if len(pokeSteps_trimmed) < 2:
                    raise Exception("Not enough Zernike mode values to calculate slope for actuator %i. "
                                    "Control matrix calculation will fail" % (ii + 1))
                    break

                # Fit a linear regression to get the relationship between actuator position and Zernike mode amplitude
                for kk in range(noZernikeModes):
                    try:
                        slopes[kk], intercepts[kk], r_values[kk], p_values[kk], std_errs[kk] = \
                            stats.linregress(pokeSteps_trimmed, zernikeModeAmp[:, kk])
                    except Exception as e:
                        print(e)

                # Input obtained slopes as the entries in the control matrix
                C_mat[:, ii] = slopes[:]
                offsets[:, ii] = intercepts[:]
                P_tests[:, ii] = p_values[:]
            else:
                print("Actuator %d is not in the pupil and therefore skipped" % (ii))
        print("Computing Control Matrix")
        self.controlMatrix = np.linalg.pinv(C_mat, rcond=threshold)
        print("Control Matrix computed")
        return self.controlMatrix

    def ac_pos_from_zernike(self, applied_z_modes, numActuators):
        if int(np.shape(applied_z_modes)[0]) < int(np.shape(self.controlMatrix)[1]):
            pad_length = int(np.shape(self.controlMatrix)[1]) - int(np.shape(applied_z_modes)[0])
            applied_z_modes = np.pad(applied_z_modes, (0,pad_length), 'constant')
        elif int(np.shape(applied_z_modes)[0]) > int(np.shape(self.controlMatrix)[1]):
            applied_z_modes = applied_z_modes[:int(np.shape(self.controlMatrix)[1])]
        else:
            pass

        actuator_pos = np.dot(self.controlMatrix, applied_z_modes)

        try:
            assert len(actuator_pos) == numActuators
        except:
            raise Exception

        return actuator_pos

    def measure_metric(self, metric_name, image, **kwargs):
        metric = metric_function[metric_name](image, **kwargs)
        return metric
    
    @staticmethod
    def find_zernike_amp_sensorless(image_stack, modes, metric_name,
                                fit_threshold=0.0, **kwargs):

        failure_flag = False
        
        metrics = []
        metric_diagnostics = []
        for image in image_stack:
            m, diag = metric_function[metric_name](image, **kwargs)
            metrics.append(m)
            metric_diagnostics.append(diag)
        metrics = np.array(metrics)

        # Fast exit for trivial cases
        if metrics.size == 1:
            return (modes[0], metrics[0]), metrics, metric_diagnostics

        #Gaussian fit formula
        def _gauss(x, A, mu, sigma, C):
            return A * np.exp(-((x - mu)**2) / (2.0 * sigma**2)) + C

        lower, upper = modes.min(), modes.max()

        # initial guesses: amp, center, width, baseline
        p0 = [
            metrics.max() - metrics.min(),
            modes[np.argmax(metrics)],
            (upper - lower) / 4.0,
            metrics.min()
        ]

        # enforce positive amplitude and positive sigma, center in [lower,upper]
        bounds = (
            [0.0, lower, 0.0,       -np.inf],
            [np.inf, upper, np.inf, np.inf]
        )

        try:
            popt, _ = curve_fit(_gauss, modes, metrics, p0=p0, bounds=bounds)
            fitted = _gauss(modes, *popt)
        except (RuntimeError, ValueError):
            print("GAUSSIAN FITTING FAILED")
            good_fit = False
            fitted = None
        else:
            #Computed R^2 coefficient to determine if the fit is good or not, based on threshold (1 is perfect, 0 is awful)
            ss_res = np.sum((metrics - fitted)**2)
            ss_tot = np.sum((metrics - metrics.mean())**2)
            r2 = 1.0 - ss_res/ss_tot if ss_tot else 0.0
            good_fit = (r2 >= fit_threshold) and (popt[0] > 0.0) #Also check the peak is positive

        if not good_fit:
            print("POOR FITTING")
            #in case of bad peak default to middle (no change)
            failure_flag = True
            mid = 0.5*(lower + upper)
            idx = np.abs(modes - mid).argmin()
            peak = (modes[idx], metrics[idx])

        else:
            A, mu, sigma, C = popt
            mu_hat = min(max(mu, lower), upper)
            peak_val = _gauss(mu_hat, A, mu, sigma, C)
            peak = (float(mu_hat), float(peak_val))

        return peak, metrics, metric_diagnostics, failure_flag

    def calc_phase_error_RMS(self, phase, modes_to_subtract=(0, 1, 2)):
        # NOTE: only works if modes_to_subtract is a contiguous subset of modes
        # Subtract modes
        modes = self.get_zernike_modes(
            phase, max(modes_to_subtract), resize_dim=128
        )
        phase_modes_to_subtract = aotools.phaseFromZernikes(
            modes,
            phase.shape[0]
        )
        phase_subtracted = phase - phase_modes_to_subtract
        # Calculate RMS
        true_flat = np.zeros(np.shape(phase_subtracted))
        return np.sqrt(
            np.mean((true_flat[self.mask] - phase_subtracted[self.mask]) ** 2)
        )
