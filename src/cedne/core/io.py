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
# ALLOWED_MODULES = [
#     "cedne",
#     "networkx"
#     "numpy",
#     "numpy.core",
#     "numpy.core.multiarray",
#     "numpy.core.numeric"
# ]

# ALLOWED_CLASSES = {
#     # Safe NumPy internals needed for unpickling arrays
#     ("numpy.core.multiarray", "_reconstruct"),
#     ("numpy.core.numeric", "_frombuffer"),
#     ("numpy", "dtype"),
#     ("numpy", "ndarray"),
#     ("numpy", "float64"),
#     ("numpy", "int64"),
#     ("builtins", "set"),
#     ("builtins", "frozenset"),
#     ("builtins", "slice"),
# }

ALLOWED_CLASSES = {
    # numpy internals
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy", "float64"),
    ("numpy", "int64"),
    # builtins
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
}

ALLOWED_MODULE_PREFIXES = [
    "cedne",
    "networkx",
    # CeDNe Worms commonly carry pandas-backed attribute tables (neurotransmitter,
    # neuropeptide, transcriptome). Pandas's pickle protocol pulls in many internal
    # classes (Series, DataFrame, BlockManager, several Index types) and the set
    # shifts across pandas minor versions, so an explicit class allowlist would be
    # brittle. We keep the wildcard for now but pair it with the DENY rules below
    # to block the known expression-evaluation gadgets (pandas.eval / pandas.query)
    # that would otherwise execute arbitrary Python during unpickling. A full
    # explicit pandas type allowlist is still on the backlog.
    "pandas",
]

# Denies override the allow-prefix above. Used to block callable gadgets that
# could trigger arbitrary code execution if reached via a crafted pickle.
DENIED_MODULE_PREFIXES = (
    "pandas.core.computation",  # pandas.eval lives here; whole subsystem evaluates expressions
    "pandas.io.pickle",  # to_pickle/read_pickle helpers — refuse chaining via unpickle
)
DENIED_QUALIFIED_NAMES = {
    ("pandas", "eval"),  # top-level re-export of pandas.core.computation.eval.eval
    ("pandas", "query"),  # similar expression-evaluation entry point
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Deny known dangerous callables first, even if they fall under an allowed prefix.
        if (module, name) in DENIED_QUALIFIED_NAMES:
            raise pickle.UnpicklingError(
                f"global '{module}.{name}' is denied (expression-evaluation gadget)"
            )
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in DENIED_MODULE_PREFIXES
        ):
            raise pickle.UnpicklingError(
                f"global '{module}.{name}' is denied (module under {DENIED_MODULE_PREFIXES})"
            )

        # Allow fully qualified safe classes
        if (module, name) in ALLOWED_CLASSES:
            return getattr(__import__(module, fromlist=[name]), name)

        # Allow all classes from whitelisted module prefixes (like cedne.*)
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in ALLOWED_MODULE_PREFIXES
        ):
            return getattr(__import__(module, fromlist=[name]), name)

        # Otherwise, reject
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")


# class RestrictedUnpickler(pickle.Unpickler):
#     """
#     A custom unpickler that restricts the loading of certain modules and classes.
#     """
#     def find_class(self, module, name):
#         # Allow all functions and classes from the allowed modules and their submodules
#         if any(module == allowed_module or module.startswith(allowed_module + ".") for allowed_module in ALLOWED_MODULES):
#             return getattr(__import__(module, fromlist=[name]), name)
#         if module == "builtins" and name in ["set", "frozenset"]:
#             return getattr(__import__(module), name)
#         raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")

# class RestrictedUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if (module, name) in ALLOWED_CLASSES:
#             return getattr(__import__(module, fromlist=[name]), name)
#         raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")


def load_pickle(file):
    """Loading restricted pickles."""
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

        with open(file_path, "rb") as pickle_file:
            # return pickle.load(pickle_file)
            w = load_pickle(pickle_file)
            if not isinstance(w, Worm):
                raise TypeError(f"Expected Worm object, got {type(w)}")
            return w
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File {file_path} not found.") from exc
    except pickle.UnpicklingError as exc:
        raise pickle.UnpicklingError(f"Failed to unpickle {file_path}.") from exc


def generate_random_string(length: int = 8) -> str:
    """
    Generates a random string of given length.

    Args:
        length (int): The length of the string to generate.

    Returns:
        str: A random string of the specified length.
    """
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


class NetworkWriter:
    """Writes the network for saving."""

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
        output_filename = "model/" + w.name + ".json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)

    def generate_json(self, w):
        """Generate a JSON-compatible dictionary from a Worm object.

        Delegates per-network serialization to ``NervousSystem.to_dict()``
        so that this writer stays in sync with the canonical serialization.

        Args:
            w: A Worm object with a ``networks`` dict of NervousSystem instances.

        Returns:
            dict: JSON-serializable representation of the worm and its networks.
        """
        json_data = {
            "model_name": w.name,
            "version": getattr(w, "version", None),
            "created_by": getattr(w, "author", None),
            "networks": {},
        }

        for net_name, network in w.networks.items():
            if hasattr(network, "to_dict"):
                json_data["networks"][net_name] = network.to_dict()

        return json_data
