"""Contains the Animal, Worm, and Fly classes for CeDNe core."""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import pickle
from .io import generate_random_string, load_pickle

class Animal:
    ''' This is a full animal class'''
    def __init__(self, species = '', name='', stage='', sex='', genotype='', **kwargs):
        ''' Initializes an Organism class'''
        self.species = species
        self.name = name
        self.stage = stage
        self.sex = sex
        self.genotype = genotype
        self.networks = {}
        for key, value in kwargs.items():
            self.set_property(key, value)


    def save(self, file_path, file_format='pickle'):
        """
        Saves the Organism object to a pickle file at the specified file path.

        Args:
            file_path (str): The path to the pickle file.
        """
        if file_format == 'pickle':
            if not file_path.endswith('.pkl'):
                file_path += '.pkl'
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(self, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        elif file_format == 'full':
            pass
        else:
            raise NotImplementedError("Only pickle format is supported.")
    
    def set_property(self, key, value):
        """
        Set a property of the organism.

        Args:
            key (str): The name of the property.
            value: The value of the property.
        """
        setattr(self, key, value)

class Worm(Animal):
    ''' This is an explicit Worm class, a container for network(s).'''
    def __init__(self, name='', stage='Day-1 Adult', sex='Hermaphrodite', genotype='N2', **kwargs) -> None:
        """
        Initializes a Worm object.

        Parameters:
            name (str): The name of the worm. If empty,
                a random alphanumeric string will be generated.
            stage (str): The stage of the worm. Default is 'Day-1 Adult'.
              Other options can be L1, L2, L3, L4, Day-2 Adult, etc.
            sex (str): The sex of the worm. Default is 'Hermaphrodite'.
                Other options can be 'Male', 'Feminized male', etc.
            genotype (str): The genotype of the worm. Default is 'N2'.
                Other options can be mutant names or other wild types, etc.

        Returns:
            None
        """
        if not name:
            name = 'Worm-' + generate_random_string()
        super().__init__(species='Caenorhabditis elegans', name=name, stage=stage, sex=sex, genotype=genotype, **kwargs)

class Fly(Animal):
    ''' This is an explicit Fly class, a container for network(s).'''
    def __init__(self, name='', stage='Day-7 Adult', sex='Female', genotype='w1118 x Canton-S G1', **kwargs) -> None:
        """
        Initializes a Fly object.

        Parameters:
            name (str): The name of the Fly. If empty,
                a random alphanumeric string will be generated.
            stage (str): The stage of the fly. Default is 'Day-7 Adult'.
              Other options can be E, L1, L2, P1, Day-2 Adult, etc.
            sex (str): The sex of the fly. Default is 'Female'.
                Other options can be 'Male', 'Feminized Male', etc.
            genotype (str): The genotype of the worm. Default is 'w1118 x Canton-S G1'.
                Other options can be mutant names or other wild-types.

        Returns:
            None
        """
        if not name:
            name = 'Fly-' + generate_random_string()
        super().__init__(species='Drosophila melanogaster', name=name, stage=stage, sex=sex, genotype=genotype)


def load_worm(file_path):
    """
    Load a Worm object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        Worm: The loaded Worm object.
    """
    try:
        with open(file_path, 'rb') as pickle_file:
            # return pickle.load(pickle_file)
            return load_pickle(pickle_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {file_path}.") from exc