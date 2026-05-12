"""
Behavioral context container for neural networks in CeDNe.

This module defines the `Behavior` class, which associates behavioral
time-series data with a neural network context. It serves as the bridge
between recorded behavioral variables (e.g., speed, heading, curvature)
and neural activity stored in `Trial`/`Session` objects.

Key class:
- `Behavior`: Stores behavioral time series, supports alignment to
  neural recordings, and links to `Session`/`Trial` for cross-referencing.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from .animal import Worm

if TYPE_CHECKING:
    from .recordings import Trial
    from .session import Session


class Behavior:
    """Behavioral data container for an organism.

    Stores named behavioral time series (e.g., speed, heading, body curvature)
    with timestamps and metadata. Can be linked to `Session` and `Trial` objects
    for alignment between behavioral and neural data.

    Attributes:
        worm: The worm object associated with this behavior.
        variables: Dict mapping variable names to 1-D numpy arrays.
        timestamps: Array of time points for the behavioral data.
        metadata: Dict of metadata (sampling_rate, units, source, etc.).
        session: Optional link to the experimental Session.
        trials: List of Trial objects linked to this behavior.
    """

    def __init__(self, worm: Worm = None, network: str = "Neutral") -> None:
        """
        Initializes a Behavior object.

        Args:
            worm: The worm object associated with the behavior.
            network: The network context name (e.g., "Neutral", "Trained").
        """
        self.worm = worm or Worm()
        self.variables: Dict[str, NDArray] = {}
        self.timestamps: Optional[NDArray] = None
        self.metadata: Dict[str, Any] = {
            "sampling_rate": None,
            "units": {},  # {"speed": "mm/s", "heading": "rad"}
            "source_file": None,
        }
        self.session: Optional["Session"] = None
        self.trials: List["Trial"] = []

        if self.worm.networks.get(network) is None:
            self.worm.networks[network] = self

    # ── Variable management ──

    def add_variable(
        self,
        name: str,
        data: NDArray,
        timestamps: Optional[NDArray] = None,
        unit: Optional[str] = None,
    ) -> None:
        """Add a named behavioral variable (1-D time series).

        Args:
            name: Variable name (e.g., "speed", "heading").
            data: 1-D array of values.
            timestamps: Optional array of time points. If provided on the
                first call, it sets the global timestamps for this Behavior.
                Subsequent calls verify length consistency.
            unit: Optional unit string (e.g., "mm/s").

        Raises:
            ValueError: If data is not 1-D or length doesn't match timestamps.
        """
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(
                f"Behavioral variable '{name}' must be 1-D, got {data.ndim}-D"
            )

        # Handle timestamps
        if timestamps is not None:
            if not isinstance(timestamps, np.ndarray):
                timestamps = np.asarray(timestamps, dtype=np.float64)
            if self.timestamps is None:
                self.timestamps = timestamps
                if self.metadata["sampling_rate"] is None and len(timestamps) > 1:
                    self.metadata["sampling_rate"] = 1.0 / np.median(
                        np.diff(timestamps)
                    )
            elif len(timestamps) != len(self.timestamps):
                raise ValueError(
                    f"Timestamps length ({len(timestamps)}) doesn't match "
                    f"existing ({len(self.timestamps)})"
                )

        if self.timestamps is not None and len(data) != len(self.timestamps):
            raise ValueError(
                f"Variable '{name}' length ({len(data)}) doesn't match "
                f"timestamps length ({len(self.timestamps)})"
            )

        self.variables[name] = data
        if unit:
            self.metadata["units"][name] = unit

    def get_variable(self, name: str) -> NDArray:
        """Get a behavioral variable by name.

        Args:
            name: Variable name.

        Returns:
            The 1-D array for the named variable.

        Raises:
            KeyError: If variable doesn't exist.
        """
        if name not in self.variables:
            raise KeyError(
                f"Behavioral variable '{name}' not found. "
                f"Available: {list(self.variables.keys())}"
            )
        return self.variables[name]

    @property
    def variable_names(self) -> List[str]:
        """List of available behavioral variable names."""
        return list(self.variables.keys())

    @property
    def n_timepoints(self) -> int:
        """Number of time points (0 if no data loaded)."""
        if self.timestamps is not None:
            return len(self.timestamps)
        if self.variables:
            return len(next(iter(self.variables.values())))
        return 0

    # ── Alignment ──

    def align_to_neural(
        self, trial: "Trial", method: str = "interpolate"
    ) -> Dict[str, NDArray]:
        """Align behavioral data to a neural recording's time base.

        Resamples all behavioral variables to match the neural trial's timestamps.

        Args:
            trial: A Trial object with recording data.
            method: Alignment method.
                - "interpolate": Linear interpolation to neural time points.
                - "nearest": Nearest-neighbor matching.

        Returns:
            Dict mapping variable names to resampled arrays matching
            the neural trial's time points.

        Raises:
            ValueError: If behavioral timestamps are not set or trial has no data.
        """
        if self.timestamps is None:
            raise ValueError(
                "Behavioral timestamps not set. Call add_variable with timestamps first."
            )

        neural_timestamps = trial.get_timestamps()
        aligned = {}

        for name, data in self.variables.items():
            if method == "interpolate":
                aligned[name] = np.interp(neural_timestamps, self.timestamps, data)
            elif method == "nearest":
                indices = np.searchsorted(
                    self.timestamps, neural_timestamps, side="left"
                )
                indices = np.clip(indices, 0, len(data) - 1)
                aligned[name] = data[indices]
            else:
                raise ValueError(
                    f"Unknown alignment method: '{method}'. "
                    f"Use 'interpolate' or 'nearest'."
                )

        return aligned

    # ── Trial/Session linkage ──

    def link_trial(self, trial: "Trial") -> None:
        """Link a Trial to this Behavior for cross-referencing.

        Also sets the trial's behavior back-reference.

        Args:
            trial: The Trial object to link.
        """
        if trial not in self.trials:
            self.trials.append(trial)
        trial.behavior = self

    def link_session(self, session: "Session") -> None:
        """Link a Session to this Behavior.

        Args:
            session: The Session object to link.
        """
        self.session = session

    # ── I/O helpers ──

    def to_dict(self) -> Dict[str, Any]:
        """Serialize behavior data to a dictionary.

        Returns:
            Dict with variables, timestamps, and metadata.
        """
        result = {
            "variables": {name: data.tolist() for name, data in self.variables.items()},
            "metadata": self.metadata,
        }
        if self.timestamps is not None:
            result["timestamps"] = self.timestamps.tolist()
        return result

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], worm: Worm = None, network: str = "Neutral"
    ) -> "Behavior":
        """Create a Behavior from a dictionary.

        Args:
            data: Dict with 'variables', optional 'timestamps', and 'metadata'.
            worm: Optional Worm object.
            network: Network context name.

        Returns:
            New Behavior instance.
        """
        behavior = cls(worm=worm, network=network)
        timestamps = None
        if "timestamps" in data:
            timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        if "metadata" in data:
            behavior.metadata.update(data["metadata"])
        for name, values in data.get("variables", {}).items():
            unit = behavior.metadata.get("units", {}).get(name)
            behavior.add_variable(
                name,
                np.asarray(values, dtype=np.float64),
                timestamps=timestamps,
                unit=unit,
            )
        return behavior

    def __repr__(self) -> str:
        return (
            f"Behavior(variables={self.variable_names}, "
            f"n_timepoints={self.n_timepoints}, "
            f"linked_trials={len(self.trials)})"
        )
