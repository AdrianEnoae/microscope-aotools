import abc
from dataclasses import dataclass
import typing
import numpy as np
from datetime import datetime
import keras as k
from microAO.aoMetrics import metric_function
from microAO.aoAlg import AdaptiveOpticsFunctions
import tifffile as tiff
import os
from microAO.pseudoPSF import pseudoPSF, make_pairs
from microAO.aoMetrics import find_noise_level
from scipy.signal.windows import tukey

@dataclass
class RoutineOutput():
    sensorless_data: dict           # Data stored between correction rounds
    done: bool = False              # Flag to indicate routine completion
    new_modes: typing.List = None   # Modes to set before next image is taken (optional)
    result: typing.Any = None       # Results returned from the routine (optional)
    status: str = None              # Status message (optional)
    error: str = None               # Error message (optional)

class Routine(metaclass=abc.ABCMeta):
    """An abstract base class for an AO routine

        A setup and image processing method must be defined.

    """

    def __init__(self, sensorless_params, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sensorless_params = sensorless_params

    @staticmethod
    @abc.abstractmethod
    def name():
        """ Return a readable name for the routine.
            eg. return 'Conventional'
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def defaults():
        """ Return dict of default parameters for the routine.
        """

    @abc.abstractmethod
    def setup(self, sensorless_data) -> dict:
        """ Perform routine setup. Returns a dict"""
        pass

    @abc.abstractmethod
    def process(self, sensorless_data) -> dict:
        """Process each image."""
        pass

@dataclass(frozen=True)
class ConventionalResults:
    metrics: typing.List
    image_stack: typing.List
    metric_diagnostics: typing.List
    modes: np.ndarray
    mode_label: str
    peak: np.ndarray = None
    failure_flag: bool = False

@dataclass(frozen=True)
class ConventionalParamsMode:
    # Noll index
    index_noll: int
    # The amplitude offsets used for scanning the mode
    offsets: np.ndarray

class ConventionalRoutine(Routine):
    def name():
        return "Conventional"

    @staticmethod
    def defaults():
        parameters = {
            "num_reps": 1,
            "NA": 1.4,
            "wavelength": 450e-9,
            "metric": 'fourier',
            "modes": (
                ConventionalParamsMode(11, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(22, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(5, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(6, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(7, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(8, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(9, np.linspace(-1.5, 1.5, 7)),
                ConventionalParamsMode(10, np.linspace(-1.5, 1.5, 7)),
            ),
            "datapoint_z": None,
            "save_as_datapoint": False,
            "log_path": None,
            'type': 'conventional'
        }

        return parameters

    def setup(self, sensorless_data):
        # Define additional data required for routine
        total_measurements = sum([len(mode.offsets) for mode in self.sensorless_params["modes"]]) * self.sensorless_params["num_reps"]
        additional_data = {
            "total_measurements": total_measurements,
            "mode_index": 0,
            "offset_index": 0,
            "correction_stack": []
        }

        # Merge additional data (note in-place merge of mutable dict)
        sensorless_data.update(additional_data)

        # Define the first correction to apply
        new_modes = sensorless_data["corrections"].copy()
        new_modes[
            self.sensorless_params["modes"][sensorless_data["mode_index"]].index_noll
            - 1
        ] += self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets[
            sensorless_data["offset_index"]
        ]

        # Update status message
        status_message = self._get_status_message(sensorless_data)

        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            status = status_message,
            new_modes = new_modes,
        )

        self.sensorless_params['fourier_noise_level']=None

        return return_data

    def process(self, sensorless_data):
        return_data = {}

        # Set default result
        result = None

        # Correct mode if enough measurements have been taken
        if sensorless_data["offset_index"] == (
            self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets.shape[0]
            - 1
        ):
            # Calculate required parameters
            mode_index_noll_0 = (
                self.sensorless_params["modes"][sensorless_data["mode_index"]].index_noll - 1
            )
            modes = (
                sensorless_data["corrections"][mode_index_noll_0]
                + self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets
            )
            image_stack = sensorless_data["image_stack"][-modes.shape[0] :]


            if self.sensorless_params["metric"]=='fourier' and self.sensorless_params['fourier_noise_level']==None:
                self.sensorless_params['fourier_noise_level']=find_noise_level(
                    image_stack[0],
                    wavelength=self.sensorless_params["wavelength"],
                    NA=self.sensorless_params["NA"],
                    pixel_size=self.sensorless_params["pixel_size"]
                    )



            # Find aberration amplitudes and correct
            peak, metrics, metric_diagnostics, failure_flag = AdaptiveOpticsFunctions.find_zernike_amp_sensorless(
                image_stack=image_stack,
                modes=modes,
                metric_name=self.sensorless_params["metric"],
                wavelength=self.sensorless_params["wavelength"],
                NA=self.sensorless_params["NA"],
                pixel_size=self.sensorless_params["pixel_size"],
                fourier_noise_level=self.sensorless_params['fourier_noise_level']
            )

            # If a peak isn't found, set abort flag
            if peak is not None:
                # Set correction (and label) in return data
                sensorless_data["corrections"][mode_index_noll_0] = peak[0]

            # Append metrics to stack
            sensorless_data["metrics_stack"].append(metrics.tolist())

            # Append current correction
            sensorless_data['correction_stack'].append(sensorless_data["corrections"].copy())

            # Instantiate result
            result = ConventionalResults(
                metrics = metrics,
                image_stack = image_stack,
                metric_diagnostics = metric_diagnostics,
                modes = modes,
                mode_label = f"Z{mode_index_noll_0 + 1}",
                peak = peak,
                failure_flag=failure_flag
            )

            # Update indices
            sensorless_data["offset_index"] = 0
            sensorless_data["mode_index"] += 1
            if sensorless_data["mode_index"] == len(self.sensorless_params["modes"]):
                sensorless_data["mode_index"] = 0

        else:
            # Increment offset index
            sensorless_data["offset_index"] += 1

        # Update status message
        status_message = self._get_status_message(sensorless_data)

        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            status = status_message,
            result = result
        )

        # Set next mode and return data, unless all measurements acquired
        if len(sensorless_data["image_stack"]) < sensorless_data["total_measurements"]:
            # Apply next set of modes
            new_modes = sensorless_data["corrections"].copy()
            new_modes[
                self.sensorless_params["modes"][sensorless_data["mode_index"]].index_noll - 1
            ] += self.sensorless_params["modes"][sensorless_data["mode_index"]].offsets[
                sensorless_data["offset_index"]
            ]

            return_data.new_modes = new_modes

        # If all data acquired, set completion flag
        else:
            return_data.done = True

        return return_data

    def _get_status_message(self, sensorless_data):
        # Update status message
        status_message = "Sensorless AO: image {n}/{N}, mode {n_mode}, meas. {n_meas}".format(
            n = len(sensorless_data["image_stack"]) + 1,
            N = sensorless_data["total_measurements"],
            n_mode = self.sensorless_params["modes"][
                sensorless_data["mode_index"]
            ].index_noll,
            n_meas = sensorless_data["offset_index"] + 1,
        )

        return status_message

class MLAOBase(Routine):
    @staticmethod
    def defaults():
        ts = datetime.strftime(datetime.now(), '%Y-%m-%d %H-%M-%S')
        log_path = f"D:/Andrei/MLAO_logs/MLWidefield-{ts}.h5"
        return {
            'n_reps': 1,
            'log_path': log_path,
            'datapoint_z': None,
            'save_as_datapoint': False,
            'type': 'MLAO'
        }

    def setup(self, sensorless_data):
        sensorless_data.update({
            "image_index": 0,
            "mode_index": 0,
            "bias_index": 0,
            "correction_stack": []
        })

        self.model = k.models.load_model(self.MODEL_PATH, compile=False)
        self.trial_modes      = self.TRIAL_MODES
        self.correction_modes = self.CORRECTION_MODES
        self.offsets          = list(self.OFFSETS)   
        self.pairs            = make_pairs(len(self.trial_modes) * len(self.offsets))

        # Initial correction
        self.initial_modes = sensorless_data["corrections"].copy()
        self.correction    = self.initial_modes.copy()

        return RoutineOutput(
            sensorless_data=sensorless_data,
            status="Initialised ML routine",
            new_modes=self.correction.copy()
        )

    def process(self, sensorless_data):
        sensorless_data['image_index'] += 1
        img_idx  = sensorless_data['image_index']
        mode_idx = sensorless_data['mode_index']

        n_trials     = len(self.trial_modes) * len(self.offsets) + 1
        total_images = self.sensorless_params['n_reps'] * n_trials + 1

        if img_idx % n_trials == 0:
            # — end of trial batch: run the network —
            start = img_idx - (len(self.trial_modes) * len(self.offsets))
            imgs  = sensorless_data['image_stack'][start:img_idx]
            arr   = np.array([im.astype('float') for im in imgs])
            _, h, w = arr.shape

            _, h, w = arr.shape
            pad_h = max(256 - h, 0)
            pad_w = max(256 - w, 0)
            if pad_h or pad_w:
                arr = self._tukey_window(arr)
                pad_top    = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left   = pad_w // 2
                pad_right  = pad_w - pad_left
                arr = np.pad(arr,((0, 0),(pad_top, pad_bottom),(pad_left, pad_right)),
                      mode='constant', constant_values=0)


            _, h, w = arr.shape
            y0, x0 = (h - 256)//2, (w - 256)//2
            arr = arr[:, y0:y0+256, x0:x0+256]

            arr = self._tukey_window(arr)

            # save raw stack
            outdir = os.path.join(os.path.expanduser("~"), "Desktop", "MLAO Image")
            os.makedirs(outdir, exist_ok=True)
            tiff.imwrite(os.path.join(outdir, "MLAO_Stack.tif"), arr)

            # prep for model
            shp = arr[0].shape
            proc = np.moveaxis(arr, 0, -1).reshape(1, shp[0], shp[1], -1).astype("float32")
            mn, mx = proc.min(), proc.max()
            proc = (proc - mn) / ((mx - mn) if mx != mn else 1e-8)

            psf_stack = pseudoPSF(proc, self.trial_modes, self.pairs, mode=self.PSEUDO_MODE)
            tiff.imwrite(os.path.join(outdir, "pseudoPSF_Stack.tif"), psf_stack)

            raw   = self.model.predict(psf_stack)[0, :]
            new   = self.correction.copy()
            for i, m in enumerate(self.correction_modes):
                new[m] -= raw[i]
            self.correction = new.copy()

            if img_idx < total_images:
                sensorless_data["corrections"]     = new.copy()
                sensorless_data['correction_stack'].append(new.copy())

            sensorless_data['mode_index'] = 0
            sensorless_data['bias_index'] = 0

        else:
            # — in‑trial step: apply bias —
            new = self.correction.copy()
            m   = self.trial_modes[mode_idx]
            new[m] += self.offsets[sensorless_data['bias_index']]
            sensorless_data['bias_index'] += 1
            if sensorless_data['bias_index'] >= len(self.offsets):
                sensorless_data['bias_index'] = 0
                sensorless_data['mode_index']  += 1

        print(f'Acquiring image ({img_idx}): {new[0:max(self.trial_modes)+1]}')

        if img_idx >= total_images:
            m = self.trial_modes[mode_idx]
            new[m] -= 1.0
            sensorless_data["corrections"]     = new.copy()
            sensorless_data['correction_stack'].append(new.copy())
            print(f'Correction applied: {new[0:max(self.correction_modes)+1]}')

        out = RoutineOutput(
            sensorless_data=sensorless_data,
            new_modes=new
        )
        if img_idx >= total_images:
            out.done = True
        return out
    
    @staticmethod
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


class ML2NDefault(MLAOBase):
    MODEL_PATH       = r'C:\microscope-aotools\models\Fullbias_Default_CorrectOrientation_32s32_savedmodel.h5'
    TRIAL_MODES      = [4, 5, 6, 7, 8, 9, 10]
    CORRECTION_MODES = TRIAL_MODES
    OFFSETS          = [1.0, -1.0]          
    PSEUDO_MODE      = 'default'

    def name():
        return "2N Default MLAO"


class ML2NWavelet(MLAOBase):
    MODEL_PATH       = r'C:\microscope-aotools\models\Fullbias_WaveletSoft_CorrectOrientation_32s32_savedmodel.h5'
    TRIAL_MODES      = [4, 5, 6, 7, 8, 9, 10]
    CORRECTION_MODES = TRIAL_MODES
    OFFSETS          = [1.0, -1.0]         
    PSEUDO_MODE      = 'wavelet'

    def name():
        return "2N Wavelet MLAO"


class MLAstgDefault(MLAOBase):
    MODEL_PATH       = r'C:\microscope-aotools\models\Astigmatism2_Default_CorrectOrientation_32s32_savedmodel.h5'
    TRIAL_MODES      = [4, 5]
    CORRECTION_MODES = [4, 5, 6, 7, 8, 9, 10]
    OFFSETS          = [1.0, -1.0]
    PSEUDO_MODE      = 'default'

    def name():
        return "Astigmatism Default MLAO"


class MLAstgWavelet(MLAOBase):
    MODEL_PATH       = r'C:\microscope-aotools\models\Astigmatism2_WaveletSoft_CorrectOrientation_32s32_savedmodel.h5'
    TRIAL_MODES      = [4, 5]
    CORRECTION_MODES = [4, 5, 6, 7, 8, 9, 10]
    OFFSETS          = [1.0, -1.0]
    PSEUDO_MODE      = 'wavelet'

    def name():
        return "Astigmatism Wavelet MLAO"






routines = {
    'conventional': ConventionalRoutine,
    '2N Default MLAO': ML2NDefault,
    '2N Wavelet MLAO': ML2NWavelet,
    'Astigmatism Default MLAO': MLAstgDefault,
    'Astigmatism Wavelet MLAO': MLAstgWavelet 
}