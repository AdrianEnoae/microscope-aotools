import abc
from dataclasses import dataclass
import typing
import numpy as np
from datetime import datetime

import tensorflow as tf

from microAO.aoMetrics import metric_function
from microAO.aoAlg import AdaptiveOpticsFunctions

from OTF_gen import Generate_OTF

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
            "NA": 1.1,
            "wavelength": 560e-9,
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
            "log_path": None
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

            # Find aberration amplitudes and correct
            peak, metrics, metric_diagnostics = AdaptiveOpticsFunctions.find_zernike_amp_sensorless(
                image_stack=image_stack,
                modes=modes,
                metric_name=self.sensorless_params["metric"],
                wavelength=self.sensorless_params["wavelength"],
                NA=self.sensorless_params["NA"],
                pixel_size=self.sensorless_params["pixel_size"],
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

class MLRoutine(Routine):
    def name():
        return "ML"

    @staticmethod
    def defaults():
        ts = datetime.strftime(datetime.now(), '%Y-%m-%d %H-%M-%S')
        log_path = f"D:/Matthew/testing/data/ML1-{ts}.h5"
        parameters = {
            'n_reps': 5,
            'log_path': log_path,
            "datapoint_z": None,
            "save_as_datapoint": False,
        }

        return parameters

    def setup(self, sensorless_data):
        # Define additional data required for routine
        additional_data = {
            "image_index": 0,
            "correction_stack": []
        }

        # Merge additional data (note in-place merge of mutable dict)
        sensorless_data.update(additional_data)

        self.model = tf.keras.models.load_model('C:/microscope-aotools/models/cdd_L1L2_var_widefield_3layer_0_8_rmse_mode8_bias_10_ast_pnve_10_12_input_2_batch32s32_savedmodel.h5', compile=False)

        # Define the first correction to apply
        self.initial_modes = sensorless_data["corrections"].copy()
        self.correction = self.initial_modes.copy()                 # Current correction (modes)
        # random_init_modes = (np.random.rand(8)-0.5)*2
        # random_init_modes = random_init_modes/np.sqrt(np.sum(random_init_modes**2))
        # random_init_modes = random_init_modes*np.random.rand()*2
        # for i, mode in enumerate([4,5,6,7,8,9,10,21]):
        #     self.correction[mode] += random_init_modes[i]
        
        # Update status message
        status_message = "Initialised ML routine"


        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            status = status_message,
            new_modes = self.correction.copy(),
        )

        return return_data

    #ANDREI'S NOTE: This is where the pseudo PSF gets computed and where the image data gets fed into the network
    def process(self, sensorless_data):
        return_data = {}

        # Set default result
        result = None

        # Update index
        sensorless_data['image_index']  += 1

        image_index = sensorless_data['image_index']


        modes_new = self.correction.copy()
        total_images = self.sensorless_params['n_reps'] * 3+1
        if image_index % 3 == 1:
            modes_new = self.correction.copy()
            modes_new[4] += 1.0
        elif image_index % 3 == 2:
            modes_new = self.correction.copy()
            modes_new[4] -= 1.0
        elif image_index % 3 == 0 :
            # Grab last 2 images
            images = sensorless_data['image_stack'][image_index-2:image_index]
            
            images_converted = np.array([image.astype('float') for image in images])
            
            images_converted = images_converted[:,128:384,128:384]
            # Get image shape
            image_shape = images_converted[0].shape
            # move stack to last dimension
            images_converted = np.moveaxis(images_converted, 0, -1)
            # Adds extra dimension   
            images_converted = images_converted.reshape(1,image_shape[0],image_shape[1],2)

            # Divide OTFs
            _, image_processed, _ = Generate_OTF(images_converted, stack_size = 2, Cropped_image_size = image_shape[1], expand_size = 1, input_image_size = 32, defocus_top = [0,1], defocus_bottom = [1,0], maxattempt=1)
            print('OTF image')

            # Predict new modes
            modes_raw = self.model.predict(image_processed)[0,:]

            # 5-10, 11, 22
            modes_new = self.correction.copy()
            for i, mode in enumerate([4,5,6,7,8,9,10,21]):
                modes_new[mode] -= modes_raw[i]

            # Store correction for next repetition
            self.correction = modes_new.copy()
            if image_index < total_images:
                sensorless_data["corrections"] = modes_new.copy()
                sensorless_data['correction_stack'].append(modes_new.copy())

        if image_index >= total_images:
            modes_new[4] -= 1.0
            sensorless_data["corrections"] = modes_new.copy()
            sensorless_data['correction_stack'].append(modes_new.copy())
        modes_new = modes_new/561*510
        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            new_modes = modes_new
        )
        if image_index >= total_images:
            return_data.done = True

        # Finish if total images acquired


        # Update status message
        status_message = "I'm running"

        # print(return_data)

        return return_data



class MLRoutine2(Routine):
    def name():
        return "ML2"

    @staticmethod
    def defaults():
        ts = datetime.strftime(datetime.now(), '%Y-%m-%d %H-%M-%S')
        log_path = f"D:/Matthew/testing/data/ML2-{ts}.h5"
        parameters = {
            'n_reps': 1,
            'log_path': log_path,
            "datapoint_z": None,
            "save_as_datapoint": False,
        }

        return parameters

    def setup(self, sensorless_data):
        # Define additional data required for routine
        additional_data = {
            "image_index": 0,
            "mode_index": 0,
            "bias_index": 0,
            "correction_stack": []
        }

        # Merge additional data (note in-place merge of mutable dict)
        sensorless_data.update(additional_data)

        self.model = tf.keras.models.load_model('C:/microscope-aotools/models/cdd_L1L2_var_widefield_3layer_0_8_rmse_mode8_bias_10_8modes_pnve_10_12_input_16_batch32s32_savedmodel.h5', compile=False)

        self.trial_modes = [4,5,6,7,8,9,10,21]


        # Define the first correction to apply
        self.initial_modes = sensorless_data["corrections"].copy()
        self.correction = self.initial_modes.copy()                 # Current correction (modes)
        # random_init_modes = (np.random.rand(8)-0.5)*2
        # random_init_modes = random_init_modes/np.sqrt(np.sum(random_init_modes**2))
        # random_init_modes = random_init_modes*np.random.rand()*2
        # for i, mode in enumerate([4,5,6,7,8,9,10,21]):
        #     self.correction[mode] += random_init_modes[i]
        # Update status message
        status_message = "Initialised ML routine"


        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            status = status_message,
            new_modes = self.correction.copy(),
        )

        return return_data

    def process(self, sensorless_data):
        return_data = {}

        # Set default result
        result = None

        # Image transforms
        print('sensorless_params', self.sensorless_params)
        # print('sensorless_data', sensorless_data)

        # Update index
        sensorless_data['image_index']  += 1

        image_index = sensorless_data['image_index']
        mode_index = sensorless_data['mode_index']

        total_images = self.sensorless_params['n_reps'] * 17+1 

        if image_index % 17 == 0 :
            # Grab last 16 images
            images = sensorless_data['image_stack'][image_index-16:image_index]
            # Get image shape

            # print('image shape:',image_shape)
            images_converted = np.array([image.astype('float') for image in images])
            images_converted = images_converted[:,128:384,128:384]
            image_shape = images_converted[0].shape
            # move stack to last dimension
            images_converted = np.moveaxis(images_converted, 0, -1)

            # print('converted image shape:',images_converted.shape)
            # Adds extra dimension   
            images_converted = images_converted.reshape(1,image_shape[0],image_shape[1],16)

            # Divide OTFs
            _, image_processed, _ = Generate_OTF(images_converted, stack_size = 16, Cropped_image_size = image_shape[1], expand_size = 1, input_image_size = 32, defocus_top = list(range(16)), defocus_bottom = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14], maxattempt=1)
            print('OTF image')

            # Predict new modes
            modes_raw = self.model.predict(image_processed)[0,:]
            print('predicted', modes_raw)
            # 5-10, 11, 22
            modes_new = self.correction.copy()
            for i, mode in enumerate([4,5,6,7,8,9,10,21]):
                modes_new[mode] -= modes_raw[i]

            # Store correction for next repetition
            self.correction = modes_new.copy()
            if image_index < total_images:
                sensorless_data["corrections"] = modes_new.copy()
                sensorless_data['correction_stack'].append(modes_new.copy())

            sensorless_data['mode_index'] = 0
            sensorless_data['bias_index'] = 0

            
        else:
            if sensorless_data['bias_index'] == 0:
                modes_new = self.correction.copy()
                modes_new[self.trial_modes[mode_index]] += 1.0

                sensorless_data['bias_index'] = 1

            else:
                modes_new = self.correction.copy()
                modes_new[self.trial_modes[mode_index]] -= 1.0

                sensorless_data['bias_index'] = 0
                sensorless_data['mode_index'] += 1

        print('current modes',image_index, modes_new)

        if image_index >= total_images:
            modes_new[self.trial_modes[mode_index]] -= 1.0
            sensorless_data["corrections"] = modes_new.copy()
            sensorless_data['correction_stack'].append(modes_new.copy())
        # Format return data
        modes_new = modes_new/561*610
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            new_modes = modes_new
        )
        if image_index >= total_images:
            return_data.done = True    
        # Finish if total images acquired


        # Update status message
        status_message = "I'm running"

        # print(return_data)

        return return_data

class MLRoutineWidefieldClarity(Routine):
    def name():
        return "ML Widefield Clarity"

    @staticmethod
    def defaults():
        ts = datetime.strftime(datetime.now(), '%Y-%m-%d %H-%M-%S')
        log_path = f"D:/Andrei/MLAO_logs/MLWidefield-{ts}.h5"
        parameters = {
            'n_reps': 1,
            'log_path': log_path,
            "datapoint_z": None,
            "save_as_datapoint": False,
        }

        return parameters

    def setup(self, sensorless_data):
        # Define additional data required for routine
        additional_data = {
            "image_index": 0,
            "mode_index": 0,
            "bias_index": 0,
            "correction_stack": []
        }

        # Merge additional data (note in-place merge of mutable dict)
        sensorless_data.update(additional_data)

        self.model = tf.keras.models.load_model('C:/microscope-aotools/models/cdd_L1L2_var_clarity_widefield_3layer_0_8_rmse_mode8_bias_10_ast_pnve_10_12_input_2_batch32s32_savedmodel.h5', compile=False)

        self.trial_modes = [4,5,6,7,8,9,10,21]


        # Define the first correction to apply
        self.initial_modes = sensorless_data["corrections"].copy()
        self.correction = self.initial_modes.copy()                 # Current correction (modes)
        # random_init_modes = (np.random.rand(8)-0.5)*2
        # random_init_modes = random_init_modes/np.sqrt(np.sum(random_init_modes**2))
        # random_init_modes = random_init_modes*np.random.rand()*2
        # for i, mode in enumerate([4,5,6,7,8,9,10,21]):
        #     self.correction[mode] += random_init_modes[i]
        # Update status message
        status_message = "Initialised ML routine"


        # Format return data
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            status = status_message,
            new_modes = self.correction.copy(),
        )

        return return_data

    def process(self, sensorless_data):
        return_data = {}

        # Set default result
        result = None

        # Image transforms
        print('sensorless_params', self.sensorless_params)
        # print('sensorless_data', sensorless_data)

        # Update index
        sensorless_data['image_index']  += 1

        image_index = sensorless_data['image_index']
        mode_index = sensorless_data['mode_index']

        total_images = self.sensorless_params['n_reps'] * 17+1 

        if image_index % 17 == 0 :
            # Grab last 16 images
            images = sensorless_data['image_stack'][image_index-16:image_index]
            # Get image shape

            # print('image shape:',image_shape)
            images_converted = np.array([image.astype('float') for image in images])
            images_converted = images_converted[:,128:384,128:384]
            image_shape = images_converted[0].shape
            # move stack to last dimension
            images_converted = np.moveaxis(images_converted, 0, -1)

            # print('converted image shape:',images_converted.shape)
            # Adds extra dimension   
            images_converted = images_converted.reshape(1,image_shape[0],image_shape[1],16)

            # Divide OTFs
            _, image_processed, _ = Generate_OTF(images_converted, stack_size = 16, Cropped_image_size = image_shape[1], expand_size = 1, input_image_size = 32, defocus_top = list(range(16)), defocus_bottom = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14], maxattempt=1)
            print('OTF image')

            # Predict new modes
            modes_raw = self.model.predict(image_processed)[0,:]
            print('predicted', modes_raw)
            # 5-10, 11, 22
            modes_new = self.correction.copy()
            for i, mode in enumerate([4,5,6,7,8,9,10,21]):
                modes_new[mode] -= modes_raw[i]

            # Store correction for next repetition
            self.correction = modes_new.copy()
            if image_index < total_images:
                sensorless_data["corrections"] = modes_new.copy()
                sensorless_data['correction_stack'].append(modes_new.copy())

            sensorless_data['mode_index'] = 0
            sensorless_data['bias_index'] = 0

            
        else:
            if sensorless_data['bias_index'] == 0:
                modes_new = self.correction.copy()
                modes_new[self.trial_modes[mode_index]] += 1.0

                sensorless_data['bias_index'] = 1

            else:
                modes_new = self.correction.copy()
                modes_new[self.trial_modes[mode_index]] -= 1.0

                sensorless_data['bias_index'] = 0
                sensorless_data['mode_index'] += 1

        print('current modes',image_index, modes_new)

        if image_index >= total_images:
            modes_new[self.trial_modes[mode_index]] -= 1.0
            sensorless_data["corrections"] = modes_new.copy()
            sensorless_data['correction_stack'].append(modes_new.copy())
        # Format return data
        modes_new = modes_new/561*610
        return_data = RoutineOutput(
            sensorless_data = sensorless_data,
            new_modes = modes_new
        )
        if image_index >= total_images:
            return_data.done = True    
        # Finish if total images acquired


        # Update status message
        status_message = "I'm running"

        # print(return_data)

        return return_data





routines = {
    'conventional': ConventionalRoutine,
    'ml': MLRoutine,
    'ml2': MLRoutine2,
    'ML Clarity Widefield': MLRoutineWidefieldClarity  
}


