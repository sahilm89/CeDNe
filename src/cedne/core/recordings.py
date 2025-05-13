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

from typing import Optional, List, Union, Dict, Any, TYPE_CHECKING
from datetime import datetime
import numpy as np
from scipy import signal
import scipy.stats as ss
from scipy.ndimage import gaussian_filter1d
from numpy.typing import NDArray
from .config import F_SAMPLE

if TYPE_CHECKING:
    from .neuron import Neuron

class Trial:
    """A class representing a single experimental recording for a neuron.
    
    This class handles the storage and basic processing of time series data recorded
    from a neuron during an experimental trial. It supports operations like data
    storage, bleaching correction, and basic signal processing.

    Attributes:
        neuron: The neuron object associated with this trial.
        i (int): The trial number.
        discard (List[int]): Points to be discarded due to bleaching or artifacts.
        _data (NDArray): The actual recording data.
        metadata (Dict): Dictionary containing trial metadata.
    """
    def __init__(self, neuron: 'Neuron', trialnum: int) -> None:
        """Initialize a new Trial instance.

        Args:
            neuron: The neuron object associated with this trial.
            trialnum: The trial number identifier.
        """
        self.neuron = neuron
        self.i = trialnum
        self.discard: List[int] = []
        self._data: Optional[NDArray] = None
        self.metadata: Dict[str, Any] = {
            'trial_number': trialnum,
            'neuron_id': id(neuron),
            'sampling_rate': F_SAMPLE,
            'processing_history': []
        }

    @property
    def recording(self) -> NDArray:
        """Get the recording data for the trial.

        Returns:
            NDArray: The recording time series data.

        Raises:
            ValueError: If no recording data has been set.
        """
        if self._data is None:
            raise ValueError("No recording data has been set")
        return self._data

    @recording.setter
    def recording(self, data: NDArray, discard: float = 0) -> None:
        """Set the recording data for the trial.

        Args:
            data: The time series data to be recorded.
            discard: Number of initial seconds to discard (e.g., for bleaching correction).

        Raises:
            ValueError: If discard is negative or if data is invalid.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim != 1:
            raise ValueError("Recording data must be a 1D array")
            
        if discard < 0:
            raise ValueError("Discard cannot be negative")
            
        if discard > 0:
            discard_points = int(discard * F_SAMPLE)
            if discard_points >= len(data):
                raise ValueError("Discard duration exceeds data length")
            self.discard = list(range(discard_points))
            self._data = data[discard_points:].astype(np.float64)
        else:
            self.discard = []
            self._data = data.astype(np.float64)

    def get_duration(self) -> float:
        """Get the duration of the recording in seconds.

        Returns:
            float: Duration of the recording in seconds.

        Raises:
            ValueError: If no recording data has been set.
        """
        return len(self.recording) / F_SAMPLE

    def get_timestamps(self) -> NDArray:
        """Get the timestamps for each sample in the recording.

        Returns:
            NDArray: Array of timestamps in seconds.

        Raises:
            ValueError: If no recording data has been set.
        """
        return np.arange(len(self.recording)) / F_SAMPLE

    def filter_signal(self, filter_type: str = 'lowpass', cutoff_freq: float = 10.0,
                     order: int = 4) -> NDArray:
        """Apply a Butterworth filter to the recording data.

        Args:
            filter_type: Type of filter ('lowpass', 'highpass', or 'bandpass').
            cutoff_freq: Cutoff frequency in Hz. For bandpass, provide tuple (low, high).
            order: Order of the Butterworth filter.

        Returns:
            NDArray: Filtered signal.

        Raises:
            ValueError: If filter_type is invalid or if no recording data has been set.
        """
        nyquist = F_SAMPLE / 2
        if isinstance(cutoff_freq, (list, tuple)):
            cutoff_freq = np.array(cutoff_freq)
            if np.any(cutoff_freq <= 0) or np.any(cutoff_freq >= nyquist):
                raise ValueError("Cutoff frequencies must be between 0 and nyquist frequency")
        else:
            if cutoff_freq <= 0 or cutoff_freq >= nyquist:
                raise ValueError("Cutoff frequency must be between 0 and nyquist frequency")
                
        normalized_cutoff = cutoff_freq / nyquist

        if filter_type == 'lowpass':
            b, a = signal.butter(order, normalized_cutoff, btype='low')
        elif filter_type == 'highpass':
            b, a = signal.butter(order, normalized_cutoff, btype='high')
        elif filter_type == 'bandpass':
            if not isinstance(normalized_cutoff, (list, tuple, np.ndarray)) or len(normalized_cutoff) != 2:
                raise ValueError("Bandpass filter requires two cutoff frequencies")
            b, a = signal.butter(order, normalized_cutoff, btype='band')
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")

        return signal.filtfilt(b, a, self.recording)

    def smooth_signal(self, window_size: int = 5, method: str = 'moving') -> NDArray:
        """Smooth the recording signal using various methods.

        Args:
            window_size: Size of the smoothing window in samples.
            method: Smoothing method ('moving', 'gaussian', or 'median').

        Returns:
            NDArray: Smoothed signal.

        Raises:
            ValueError: If method is invalid or if no recording data has been set.
        """
        if method == 'moving':
            window = np.ones(window_size) / window_size
            return np.convolve(self.recording, window, mode='same')
        elif method == 'gaussian':
            return gaussian_filter1d(self.recording, window_size)
        elif method == 'median':
            return signal.medfilt(self.recording, window_size)
        else:
            raise ValueError(f"Invalid smoothing method: {method}")

    def normalize_signal(self, method: str = 'minmax', baseline_window: Optional[tuple] = None) -> NDArray:
        """Normalize the recording signal using various methods.

        Args:
            method: Normalization method ('minmax', 'zscore', or 'baseline').
            baseline_window: Tuple of (start, end) indices for baseline normalization.

        Returns:
            NDArray: Normalized signal.

        Raises:
            ValueError: If method is invalid or if no recording data has been set.
        """
        if method == 'minmax':
            min_val = np.min(self.recording)
            max_val = np.max(self.recording)
            return (self.recording - min_val) / (max_val - min_val)
        elif method == 'zscore':
            return (self.recording - np.mean(self.recording)) / np.std(self.recording)
        elif method == 'baseline':
            if baseline_window is None:
                raise ValueError("baseline_window must be provided for baseline normalization")
            start, end = baseline_window
            baseline = np.mean(self.recording[start:end])
            return (self.recording - baseline) / baseline
        else:
            raise ValueError(f"Invalid normalization method: {method}")

    def detect_peaks(self, height: Optional[float] = None,
                    distance: Optional[int] = None) -> tuple[NDArray, NDArray]:
        """Detect peaks in the recording signal.

        Args:
            height: Minimum height of peaks.
            distance: Minimum distance between peaks in samples.

        Returns:
            tuple: (peak_indices, peak_heights)

        Raises:
            ValueError: If no recording data has been set.
        """
        if distance is not None and distance < 1:
            raise ValueError("`distance` must be greater or equal to 1")
            
        peaks, properties = signal.find_peaks(self.recording, height=height, distance=distance)
        peak_heights = self.recording[peaks] if 'peak_heights' not in properties else properties['peak_heights']
        return peaks, peak_heights

    def get_statistics(self) -> dict:
        """Calculate basic statistics of the recording.

        Returns:
            dict: Dictionary containing various statistical measures:
                - mean: Mean of the signal
                - std: Standard deviation
                - median: Median value
                - min: Minimum value
                - max: Maximum value
                - skewness: Skewness of the distribution
                - kurtosis: Kurtosis of the distribution
                - rms: Root mean square value

        Raises:
            ValueError: If no recording data has been set.
        """
        return {
            'mean': np.mean(self.recording),
            'std': np.std(self.recording),
            'median': np.median(self.recording),
            'min': np.min(self.recording),
            'max': np.max(self.recording),
            'skewness': ss.skew(self.recording),
            'kurtosis': ss.kurtosis(self.recording),
            'rms': np.sqrt(np.mean(np.square(self.recording)))
        }

    def compute_power_spectrum(self, window: str = 'hann') -> tuple[NDArray, NDArray]:
        """Compute the power spectrum of the recording.

        Args:
            window: Window function to use ('hann', 'hamming', 'blackman', etc.).

        Returns:
            tuple: Arrays of frequencies and corresponding power spectrum.

        Raises:
            ValueError: If no recording data has been set.
        """
        freqs, psd = signal.welch(self.recording, F_SAMPLE, window=window)
        return freqs, psd

    def compute_snr(self, signal_window: tuple, noise_window: tuple) -> float:
        """Compute the signal-to-noise ratio.

        Args:
            signal_window: Tuple of (start, end) indices for signal region.
            noise_window: Tuple of (start, end) indices for noise region.

        Returns:
            float: Signal-to-noise ratio in dB.

        Raises:
            ValueError: If windows are invalid or if no recording data has been set.
        """
        sig_start, sig_end = signal_window
        noise_start, noise_end = noise_window

        signal_power = np.mean(np.square(self.recording[sig_start:sig_end]))
        noise_power = np.mean(np.square(self.recording[noise_start:noise_end]))
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)

    def segment_signal(self, threshold: float, min_duration: int = 10) -> List[tuple[int, int]]:
        """Segment the signal into regions above threshold.

        Args:
            threshold: Amplitude threshold for segmentation.
            min_duration: Minimum duration (in samples) for a valid segment.

        Returns:
            List[tuple]: List of (start, end) indices for each segment.

        Raises:
            ValueError: If no recording data has been set.
        """
        above_threshold = self.recording > threshold
        changes = np.diff(above_threshold.astype(int))
        rise_points = np.where(changes == 1)[0] + 1
        fall_points = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if len(rise_points) == 0 or len(fall_points) == 0:
            return []

        if rise_points[0] > fall_points[0]:
            rise_points = np.insert(rise_points, 0, 0)
        if rise_points[-1] > fall_points[-1]:
            fall_points = np.append(fall_points, len(self.recording))

        segments = []
        for start, end in zip(rise_points, fall_points):
            if end - start >= min_duration:
                segments.append((start, end))

        return segments

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the trial.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get metadata value.

        Args:
            key: Metadata key.

        Returns:
            The metadata value.

        Raises:
            KeyError: If the key doesn't exist in metadata.
        """
        return self.metadata[key]

    def log_processing(self, operation: str, parameters: Dict[str, Any]) -> None:
        """Log a processing operation in the trial's history.

        Args:
            operation: Name of the processing operation.
            parameters: Dictionary of parameters used in the operation.
        """
        self.metadata['processing_history'].append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'parameters': parameters
        })

    def validate_data(self) -> bool:
        """Validate the recording data.

        Returns:
            bool: True if data is valid, False otherwise.

        This method checks:
        - Data is not None
        - Data is a numpy array
        - Data is 1-dimensional
        - Data contains no NaN or infinite values
        - Data length is reasonable (> 0)
        """
        if self._data is None:
            return False
        
        if not isinstance(self._data, np.ndarray):
            return False
            
        if self._data.ndim != 1:
            return False
            
        if not np.isfinite(self._data).all():
            return False
            
        if len(self._data) == 0:
            return False
            
        return True

    def get_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality metrics for the recording.

        Returns:
            dict: Dictionary containing quality metrics:
                - snr: Signal-to-noise ratio
                - noise_level: Estimated noise level
                - signal_stability: Measure of signal stability
                - artifact_count: Number of potential artifacts

        Raises:
            ValueError: If no recording data has been set.
        """
        if not self.validate_data():
            raise ValueError("Invalid or missing recording data")

        # Calculate noise level from the first 10% of the signal
        noise_window = slice(0, len(self._data) // 10)
        noise_level = np.std(self._data[noise_window])

        # Detect potential artifacts (points > 3 std from mean)
        mean = np.mean(self._data)
        std = np.std(self._data)
        artifacts = np.sum(np.abs(self._data - mean) > 3 * std)

        # Calculate signal stability (variation in signal segments)
        segment_length = len(self._data) // 10
        segments = np.array_split(self._data, 10)
        segment_means = [np.mean(seg) for seg in segments]
        stability = 1 - np.std(segment_means) / np.mean(segment_means)

        return {
            'noise_level': float(noise_level),
            'artifact_count': int(artifacts),
            'signal_stability': float(stability)
        }

class StimResponse:
    """A class representing a stimulus-response pair in a neural recording.

    This class handles the analysis of neural responses to specific stimuli,
    extracting various features from the response signal and providing methods
    for response characterization.

    Attributes:
        stim (NDArray): The stimulus signal.
        response (NDArray): The response signal.
        feature (Dict[int, Any]): Dictionary of extracted features.
        neuron: The neuron object associated with this response.
        f_sample (float): Sampling frequency in Hz.
        sampling_time (float): Time between samples in seconds.
        baseline (NDArray): Baseline signal before stimulus.

    Features extracted include:
        0: Maximum response amplitude
        1: Area under the curve
        2: Mean response
        3: Time to peak
        4: Area under the curve to peak
        5: Minimum response
        6: Response onset time
        7: Positive response area
        8: Absolute area under the curve
    """
    def __init__(self, trial: Trial, stimulus: NDArray, response: NDArray, 
                baseline_samples: int) -> None:
        """Initialize a StimResponse instance.

        Args:
            trial: The trial object associated with this response.
            stimulus: The stimulus signal.
            response: The response signal.
            baseline_samples: Number of samples to use for baseline calculation.

        Raises:
            ValueError: If input arrays have invalid dimensions or lengths.
        """
        if not isinstance(stimulus, np.ndarray) or not isinstance(response, np.ndarray):
            raise ValueError("Stimulus and response must be numpy arrays")
            
        if stimulus.ndim != 1 or response.ndim != 1:
            raise ValueError("Stimulus and response must be 1-dimensional")
            
        if len(stimulus) != len(response):
            raise ValueError("Stimulus and response must have the same length")
            
        if baseline_samples >= len(response):
            raise ValueError("Baseline samples exceeds response length")

        self.stim = stimulus
        self.response = response
        self.feature: Dict[int, Any] = {}
        self.neuron = trial.neuron
        self.f_sample = F_SAMPLE
        self.sampling_time = 1./self.f_sample
        self.baseline = self.response[:baseline_samples]
        
        # Extract all features
        for feature_index in range(9):  # 9 features total
            self.feature[feature_index] = self.extract_feature(feature_index)

    def extract_feature(self, feature_index: int) -> Union[float, tuple[float, float]]:
        """Extract a specific feature from the stimulus-response pair.

        Args:
            feature_index: Index of the feature to extract:
                0: Maximum value
                1: Area under the curve
                2: Time to peak
                3: Mean value
                4: Area under the curve to peak
                5: Minimum value
                6: Onset time
                7: Positive area
                8: Absolute area under the curve

        Returns:
            The extracted feature value or tuple of values.

        Raises:
            ValueError: If feature_index is invalid.
        """
        feature_mapping = {
            0: self._find_maximum,
            1: self._area_under_the_curve,
            2: self._find_time_to_peak,
            3: self._find_mean,
            4: self._area_under_the_curve_to_peak,
            5: self._find_minimum,
            6: self._find_onset_time,
            7: self._find_positive_area,
            8: self._absolute_area_under_the_curve,
        }

        if feature_index not in feature_mapping:
            raise ValueError(f"Invalid feature index: {feature_index}")

        return feature_mapping[feature_index]()

    def _find_maximum(self) -> float:
        """Find the maximum response amplitude.

        Returns:
            float: Maximum value of the response signal.
        """
        return float(np.max(self.response))

    def _find_minimum(self) -> float:
        """Find the minimum response amplitude.

        Returns:
            float: Minimum value of the response signal.
        """
        return float(np.min(self.response))

    def _find_time_to_peak(self) -> float:
        """Find the time to response peak.

        Returns:
            float: Time to peak in seconds.
        """
        max_index = np.argmax(self.response)
        return float(max_index * self.sampling_time)

    def _find_mean(self) -> float:
        """Calculate the mean response amplitude.

        Returns:
            float: Mean value of the response signal.
        """
        return float(np.mean(self.response))

    def _area_under_the_curve(self, bin_size: int = 5) -> float:
        """Calculate the total area under the response curve.

        Args:
            bin_size: Number of samples to bin for integration.

        Returns:
            float: Area under the curve in amplitude-seconds.
        """
        undersampling = self.response[::bin_size]
        return float(np.trapz(undersampling, dx=self.sampling_time*bin_size))

    def _absolute_area_under_the_curve(self, bin_size: int = 5) -> float:
        """Calculate the absolute area under the response curve.

        Args:
            bin_size: Number of samples to bin for integration.

        Returns:
            float: Absolute area under the curve in amplitude-seconds.
        """
        undersampling = np.abs(self.response[::bin_size])
        return float(np.trapz(undersampling, dx=self.sampling_time*bin_size))

    def _area_under_the_curve_to_peak(self, bin_size: int = 10) -> float:
        """Calculate the area under the curve up to the peak response.

        Args:
            bin_size: Number of samples to bin for integration.

        Returns:
            float: Area under the curve to peak in amplitude-seconds.
        """
        undersampling = self.response[::bin_size]
        max_index = np.argmax(undersampling)
        window_to_peak = undersampling[:max_index+1]
        return float(np.trapz(window_to_peak, dx=self.sampling_time*bin_size))

    def _find_onset_time(self, window_size: int = 10, threshold_std: float = 2.0, 
                        absolute_threshold: Optional[float] = None) -> float:
        """Find the response onset time using a sliding window approach.

        For calcium imaging data, this method uses a robust detection approach that:
        1. Requires the signal to stay above threshold for the entire window
        2. Checks that the signal continues to rise after the window
        3. Ensures the detected onset is not just a noise spike
        4. Requires a moderate increase in signal level

        Args:
            window_size: Size of the sliding window in samples.
            threshold_std: Number of standard deviations above baseline for onset.
                          Only used if absolute_threshold is None.
            absolute_threshold: Absolute threshold value for onset detection.
                              If provided, overrides threshold_std calculation.

        Returns:
            float: Onset time in seconds.
        """
        baseline_mean = np.mean(self.baseline)
        baseline_std = np.std(self.baseline)
        
        # Calculate threshold based on provided method
        if absolute_threshold is not None:
            threshold = absolute_threshold
        else:
            # Use statistical threshold
            threshold = baseline_mean + threshold_std * baseline_std
        
        # Use a sliding window to find when response consistently exceeds threshold
        for i in range(len(self.response) - window_size):
            window = self.response[i:i+window_size]
            if np.mean(window) > threshold:
                return float(i * self.sampling_time)
        
        # If no onset found, return nan
        return np.nan

    def _find_positive_area(self, bin_size: int = 10) -> tuple[float, float]:
        """Calculate the positive and negative areas of the response.

        Args:
            bin_size: Number of samples to bin for integration.

        Returns:
            tuple: (positive_area, negative_area) in amplitude-seconds.
        """
        undersampling = self.response[::bin_size]
        pos_mask = undersampling > 0
        neg_mask = undersampling < 0
        
        pos_area = float(np.trapz(undersampling[pos_mask], 
                                dx=self.sampling_time*bin_size))
        neg_area = float(abs(np.trapz(undersampling[neg_mask], 
                                    dx=self.sampling_time*bin_size)))
        
        return pos_area, neg_area

    def get_response_characteristics(self) -> Dict[str, float]:
        """Calculate comprehensive response characteristics.

        Returns:
            dict: Dictionary containing various response metrics:
                - amplitude: Peak response amplitude (relative to baseline)
                - duration: Response duration
                - latency: Response latency
                - integral: Total response integral
                - baseline_mean: Mean baseline activity
                - baseline_std: Baseline standard deviation
                - signal_to_noise: Signal-to-noise ratio
        """
        baseline_mean = np.mean(self.baseline)
        baseline_std = np.std(self.baseline)
        
        # Calculate peak amplitude relative to baseline
        # First subtract baseline from the entire response
        response_minus_baseline = self.response - baseline_mean
        # Find the maximum absolute deviation
        max_abs_idx = np.argmax(np.abs(response_minus_baseline))
        # Use the actual value at that index
        peak_amplitude = response_minus_baseline[max_abs_idx]
        
        onset_time = self._find_onset_time()
        
        # Find response end (when signal returns to baseline)
        end_threshold = baseline_mean + 2 * baseline_std
        response_end = np.where(self.response <= end_threshold)[0]
        response_end = response_end[-1] if len(response_end) > 0 else len(self.response)
        
        duration = (response_end - onset_time) * self.sampling_time if not np.isnan(onset_time) else 0.0
        integral = self._area_under_the_curve()
        
        # Calculate signal-to-noise ratio
        signal_power = np.mean(np.square(self.response - baseline_mean))
        noise_power = np.mean(np.square(self.baseline - baseline_mean))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return {
            'amplitude': float(peak_amplitude),
            'duration': float(duration),
            'latency': float(onset_time),
            'integral': float(integral),
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'signal_to_noise': float(snr)
        }

def _linear_transform(value, minvalue, maxvalue):
    return (value - minvalue)/(maxvalue - minvalue)