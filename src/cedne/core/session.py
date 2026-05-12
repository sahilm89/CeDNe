"""
Session and experimental event tracking for CeDNe.

This module defines the Session class, which acts as a container for experimental
trials and links them to specific metadata and context.
"""

from typing import Dict, List, Optional
from .recordings import Trial
from .context import Context


class Session:
    """
    Represents an experimental recording session.

    A session groups multiple trials together and associates them with
    experimental metadata (date, experimenter, lab) and an experimental context.
    """

    def __init__(self, name: str, context: Optional[Context] = None, **kwargs):
        """
        Initializes a new Session instance.

        Args:
            name (str): Unique name/identifier for the session.
            context (Context, optional): The experimental context for this session.
            **kwargs: Additional metadata (e.g., date, experimenter, lab, notes).
        """
        self.name = name
        self.context = context
        self.trials: Dict[
            int, List[Trial]
        ] = {}  # Maps trial numbers to list of Trials (across neurons)

        # Store additional metadata
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_trial(self, trial: Trial):
        """
        Adds a trial to the session.
        """
        if trial.i not in self.trials:
            self.trials[trial.i] = []
        self.trials[trial.i].append(trial)

    def __str__(self):
        return f"Session(name='{self.name}', context='{self.context.name if self.context else None}', num_trials={len(self.trials)})"
