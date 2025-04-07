"""
Trial-wise neural activity and stimulus-response analysis for CeDNe.

This module defines classes for capturing and analyzing experimental recordings 
from neurons. It includes:

- `Trial`: Represents a single experimental recording for a neuron, such as 
  a calcium imaging time series. Trials are stored per-neuron in the `Neuron.trial` 
  dictionary and support signal preprocessing (e.g., bleaching correction).
  
- `StimResponse`: Encapsulates a stimulus-response pair recorded during a `Trial`,
  and extracts a set of interpretable features from the response signal, including
  max amplitude, onset time, area under the curve, and others.

These classes are designed to support time-locked calcium imaging experiments 
and help link dynamic neural activity to behavioral or stimulus-driven contexts.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import numpy as np
from scipy import signal
import scipy.stats as ss


class Trial:
    """ This is the trial class for different trials on the same wo Write a utir, neuron, etc"""
    def __init__(self, parent, trialnum):
        """
        Initializes the Trial object with the given parent and trial number.

        Parameters:
            parent (datatype): 
                Description of the parameter.
            trialNum (datatype): 
                Description of the parameter.

        Returns:
            None
        """
        self.parent = parent
        self.i = trialnum

    @property
    def recording(self):
        """
        Get the recording data for the Trial object.

        Returns:
            datatype: The recording data.
        """
        return self._data

    @recording.setter
    def recording(self, _data, discard=0):
        """
        Set the recording data for the Trial object.

        Parameters:
            signal (array-like): The timecourse signal to be recorded.

        Raises:
            ValueError: If the length of the signal is not 451 or 601.

        Returns:
            None
        """
        if not discard:
            self.discard = []
            self._data = _data.astype(np.float64)
        elif discard>0:
            self.discard = discard*F_SAMPLE #Initial points to be discarded due to bleaching, etc.
            self._data = _data[discard*F_SAMPLE:].astype(np.float64)
        else:
            raise ValueError("Discard cannot be negative")

class StimResponse:
    """
    This is the stimulus and response class, for different trials on the same worm, neuron, etc
    """
    def __init__(self, trial, stimulus, response, baseline) -> None:
        """
        Initializes a StimResponse object.

        Parameters:
            trial (Trial): 
                The trial object associated with the stimulus and response.
            stimulus (array-like): 
                The stimulus signal.
            response (array-like): 
                The response signal.
            baseline (int): 
                The number of baseline samples to consider for the response.

        Returns:
            None
        """
        self.stim = stimulus
        self.response = response
        self.feature = {}
        self.neuron = trial.parent
        self.f_sample = F_SAMPLE # frames per sec
        self.sampling_time = 1./self.f_sample
        self.baseline = self.response[:baseline]

        for feature_index in range(len(self.neuron.features)):
            self.feature.update({feature_index: self.extract_feature(feature_index)})

    def extract_feature(self, feature_index):
        """
        Extracts a specific feature from the stimulus response pair.

        Parameters:
            feature_index (int): The index of the feature to extract. Possible values are:
                0: 
                    Maximum value of the response.
                1: 
                    Area under the curve of the response.
                2: 
                    Mean value of the response.
                3: 
                    Time to peak of the response.
                4: 
                    Area under the curve to peak of the response.
                5: 
                    Minimum value of the response.
                6: 
                    Onset time of the response.
                7: 
                    Positive area of the response.
                8: 
                    Absolute area under the curve of the response.

        Returns:
            The extracted feature value. The return type depends on the feature index.
        """

        feature_mapping = {
            0: self._find_maximum,
            1: self._area_under_the_curve,
            2: self._find_mean,
            3: self._find_time_to_peak,
            4: self._area_under_the_curve_to_peak,
            5: self._find_minimum,
            6: self._find_onset_time,
            7: lambda: (self._find_positive_area()[0], self._find_positive_area()[1]),
            8: self._absolute_area_under_the_curve,
        }

        return feature_mapping[feature_index]()

    # Features
    def _find_maximum(self):
        '''Finds the maximum of the vector in a given interest'''
        return np.max(self.response)

    def _find_minimum(self):
        '''Finds the maximum of the vector in a given interest'''
        return np.min(self.response)

    def _find_time_to_peak(self):
        '''Finds the time to maximum of the vector in a given interest'''
        max_index = np.argmax(self.response)
        time_to_peak = (max_index)*self.sampling_time
        return time_to_peak

    def _find_mean(self):
        '''Finds the mean of the vector in a given interest'''
        return np.average(self.response)

    def _area_under_the_curve(self, bin_size=5):
        '''Finds the area under the curve of the vector in the given window.
           This will subtract negative area from the total area.'''
        undersampling = self.response[::bin_size]
        auc = np.trapz(undersampling, dx=self.sampling_time*bin_size)  # in V.s
        return auc

    def _absolute_area_under_the_curve(self, bin_size=5):
        '''Finds the area under the curve of the vector in the given window.
           This will subtract negative area from the total area.'''
        undersampling = np.abs(self.response[::bin_size])
        auc = np.trapz(undersampling, dx=self.sampling_time*bin_size)  # in V.s
        return auc

    def _area_under_the_curve_to_peak(self, bin_size=10):
        '''Finds the area under the curve of the vector in the given window'''
        undersampling = self.response[::bin_size]
        max_index = np.argmax(undersampling)
        window_to_peak = undersampling[:max_index+1]
        auctp = np.trapz(window_to_peak, dx=self.sampling_time*bin_size)  # in V.s
        return auctp

    def _find_onset_time(self, step=2, slide = 1, init_pval_tolerance=0.5):
        ''' Find the onset of the curve using a 2 sample KS test
	maxOnset, step and slide are in ms'''

        window_size = int(step*self.f_sample)
        step_size = int(slide*self.f_sample)
        index_right = window_size
        for index_left in range(0, len(self.response)+1-window_size, step_size):
            _stat, pval = ss.ks_2samp(self.baseline, self.response[index_left:index_right])
            index_right += step_size
            if pval < init_pval_tolerance:
                # print index_left, pVal, stat#, self.self.response_raw[index_left:index_right]
                break
        return float(index_left)/self.f_sample

    def _find_positive_area(self, bin_size=10):
        '''Finds the area under the curve of the vector in the given window'''
        undersampling = self.response[::bin_size]
        undersampling = undersampling[np.where(undersampling>0.)]
        auc_pos = np.trapz(undersampling,  dx=self.sampling_time*bin_size)  # in V.s
        positive_time = self.sampling_time*len(undersampling)
        return auc_pos, positive_time

    #def _flagNoise(self, pValTolerance=0.01):
    #    ''' This function asseses if the distributions of the baseline
    #    and interest are different or not '''
    #    m, pVal = ss.ks_2samp(self.baselineWindow, self.self.response)
    #    if pVal < pValTolerance:
    #        return 0
    #    else:
    #        print "No response measured in trial {}".format(self.index)
    #        return 1  # Flagged as noisy

    def _normalize_to_baseline(self, baseline_window):
        '''normalizes the vector to an average baseline'''
        baseline = np.average(baseline_window)
        response_new = self.response - baseline  # Subtracting baseline from whole array
        return response_new, baseline

    def _time_series_filter(self, ts_filter='', cutoff=2000., order=4, trace=None):
        ''' Filter the time series vector '''
        if trace is None:
            trace = self.response
        if ts_filter == 'bessel': # f_sample/2 is Niquist, cutoff is the low-pass cutoff.
            cutoff_to_niquist_ratio = 2*cutoff/(self.f_sample)
            _b, _a = signal.bessel(order, cutoff_to_niquist_ratio, analog=False)
            trace =  signal.filtfilt(_b, _a, trace)
        return trace

    def _smoothen(self, smoothening_time):
        '''normalizes the vector to an average baseline'''
        smoothening_window = smoothening_time*self.f_sample
        window = np.ones(int(smoothening_window)) / float(smoothening_window)
        self.response = np.convolve(self.response, window, 'same')  # Convolving with a rectangle
        return self.response

def _linear_transform(value, minvalue, maxvalue):
    return (value - minvalue)/(maxvalue - minvalue)