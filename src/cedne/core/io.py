"""Contains I/O helpers for loading pickles and worms."""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import pickle
import random
import string
import json
# import py2neo

# Restricting unpickling
ALLOWED_MODULES = [
    "cedne",
    "networkx"
]

class RestrictedUnpickler(pickle.Unpickler):
    """
    A custom unpickler that restricts the loading of certain modules and classes.
    """
    def find_class(self, module, name):
        # Allow all functions and classes from the allowed modules and their submodules
        if any(module.startswith(allowed_module) for allowed_module in ALLOWED_MODULES):
            return getattr(__import__(module, fromlist=[name]), name)
        if module == "builtins" and name in ["set", "frozenset"]:
            return getattr(__import__(module), name)
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")

def load_pickle(file):
    ''' Loading restricted pickles.'''
    return RestrictedUnpickler(file).load()

def load_worm(file_path):
    """
    Load a Worm object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        Worm: The loaded Worm object.
    """
    try:
        from .animal import Worm
        with open(file_path, 'rb') as pickle_file:
            # return pickle.load(pickle_file)
            w= load_pickle(pickle_file)
            if not isinstance(w, Worm):
                raise TypeError(f"Expected Worm object, got {type(w)}")
            return w
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File {file_path} not found.") from exc
    except pickle.UnpicklingError as exc:
        raise pickle.UnpicklingError(f"Failed to unpickle {file_path}.") from exc
    except TypeError as exc:
        raise TypeError(f"Expected Worm object, got {type(w)}") from exc
    
def generate_random_string(length: int = 8) -> str:
    """
    Generates a random string of given length.

    Args:
        length (int): The length of the string to generate.

    Returns:
        str: A random string of the specified length.
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

class NetworkWriter:
    ''' Writes the network for saving.'''
    def __init__(self):
        """
        Initializes the NetworkWriter object.
        Placeholder for future functionality."""
        NEO4J_URI = "bolt://localhost:7687"
        NEO4J_USER = "neo4j"
        NEO4J_PASS = "password"
        OUTPUT_JSON = "generated_metadata.json"
        # Connect to Neo4j database
        # graph_db = py2neo.Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    def write(self, w):
        json_data = self.generate_json(w)
        output_filename = 'model/' + w.name + '.json'
        with open(output_filename, "w", encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

    def generate_json(self, w):
        json_data = {
            "model_name": w.name,
            "version": w.version,
            "created_by": w.author,
            "date_created": "2025-02-10",
            "networks": {},
            "neurons": {},
            "connections": [],
            "neo4j_query": "MATCH (n:Neuron)-[r:SYNAPSE]->(m:Neuron) WHERE n.type = 'Sensory' RETURN n, r, m"
        }

        # Store neurons and their properties
        for nn in w.networks:
            for n in nn.neurons:
                json_data["neurons"][n] = {
                    #"data":  
                    }

            # Store edges (synapses)
            for u, v, data in nn.edges(data=True):
                json_data["connections"].append({
                    "source": u.name,
                    "target": v.name,
                    "weight": data.get("weight", 1.0),
                    #"neurotransmitters": data.get("neurotransmitters", [])
                })

        return json_data