"""
Behavioral context container for neural networks in CeDNe.

This module defines the `Behavior` class, which associates a `Worm` object 
with a specific network (e.g., "Neutral", "Trained") in a behavioral context.

Key class:
- `Behavior`: A lightweight wrapper used to annotate or tag a worm's network 
  with behavioral meaning. It serves as a placeholder for integrating future 
  behavioral analysis, but currently stores no trial data or time-series information.

Intended Use:
This class is useful for organizing multiple networks per worm, such as 
baseline vs. trained conditions, or different environmental contexts.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

from .animal import Worm 
class Behavior:
    ''' This is a behavior class for the organism'''
    def __init__(self, worm: Worm = None, network: str = "Neutral") -> None:
        """
        Initializes a Behavior object.

        Args:
            worm (Worm, optional): The worm object associated with the behavior. Defaults to None.
            network (str, optional): The network for the behavior. Defaults to "Neutral".
        """
        self.worm = worm or Worm()
        if self.worm.networks.get(network) is None:
            self.worm.networks[network] = self