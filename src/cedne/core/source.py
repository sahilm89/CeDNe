"""Provenance and citation infrastructure for CeDNe.
"""

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple


__author__ = "Sahil Moza"
__date__ = "2026-04-27"
__license__ = "MIT"


@dataclass
class Citation:
    """A bibliographic reference: paper, dataset, or other evidence source.

    The ``key`` is the canonical identifier (BibTeX-style, e.g. ``"White1986"``).
    All other fields are optional; populate as much metadata as is available.
    """

    key: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict, omitting None-valued fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Citation":
        """Build a Citation from a dict; ignores unknown keys."""
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


def serialize_citations(citations: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a citations dict to JSON-friendly form.

    Citation instances are flattened via ``Citation.to_dict``; any other value
    (e.g. legacy loader dicts, strings) is passed through unchanged.
    """
    out: Dict[str, Any] = {}
    for key, val in citations.items():
        if isinstance(val, Citation):
            out[key] = val.to_dict()
        else:
            out[key] = val
    return out


class Citable:
    """Base class adding a citations container and hierarchical resolution.

    Subclasses should call ``Citable.__init__(self)`` somewhere in their own
    ``__init__`` (the call is idempotent, so order with other base classes
    doesn't matter). Subclasses may override ``_parent_citables`` to define
    which container objects ``effective_citations`` should walk into.

    The ``citations`` attribute is a plain dict. Values may be ``Citation``
    instances OR arbitrary user data (legacy loaders attach plain dicts).
    The hierarchical resolution treats both uniformly.
    """

    def __init__(self) -> None:
        # Idempotent: don't clobber an existing citations dict.
        if not hasattr(self, "citations") or self.citations is None:
            self.citations: Dict[str, Any] = {}

    def add_citation(self, citation: "Citation", key: Optional[str] = None) -> None:
        """Attach a structured Citation. Defaults the dict key to citation.key."""
        if key is None:
            key = citation.key
        # Lazy-init if a subclass forgot to call Citable.__init__
        if not hasattr(self, "citations") or self.citations is None:
            self.citations = {}
        self.citations[key] = citation

    def remove_citation(self, key: str) -> None:
        """Remove a citation by dict key. No-op if absent."""
        if hasattr(self, "citations") and self.citations:
            self.citations.pop(key, None)

    def _parent_citables(self) -> Iterable["Citable"]:
        """Override to return parent citable containers.

        Default: no parents (terminates the walk). Each subclass defines its
        own walk: a Neuron's parents are the NeuronGroups that contain it plus
        its NervousSystem; a NervousSystem's parent is its Animal; etc.
        """
        return ()

    def _provenance_label(self) -> str:
        """Human-readable label for this object's level in the hierarchy."""
        name = getattr(self, "name", None) or getattr(self, "group_name", None)
        if name:
            return f"{type(self).__name__}({name})"
        return type(self).__name__

    def citations_to_dict(self) -> Dict[str, Any]:
        """Serialize the local citations dict to JSON-friendly form.

        Citation instances are flattened via ``Citation.to_dict``; any other
        value (e.g. legacy loader dicts) is passed through as-is.
        """
        return serialize_citations(self.citations) if hasattr(self, "citations") else {}

    def effective_citations(
        self, _visited: Optional[set] = None
    ) -> List[Tuple[str, str, Any]]:
        """Walk up the containment hierarchy and return all applicable citations.

        Returns:
            A list of ``(provenance_label, citation_key, citation_value)``
            tuples. The label identifies which object in the hierarchy carried
            the citation (most-specific first). Duplicate citation_keys are
            dropped, keeping the most-specific attribution.

            ``citation_value`` may be a ``Citation`` instance or any other
            value the caller stored under that key.
        """
        if _visited is None:
            _visited = set()
        if id(self) in _visited:
            return []
        _visited.add(id(self))

        seen_keys: set = set()
        result: List[Tuple[str, str, Any]] = []

        my_label = self._provenance_label()
        if hasattr(self, "citations") and self.citations:
            for key, val in self.citations.items():
                if key not in seen_keys:
                    result.append((my_label, key, val))
                    seen_keys.add(key)

        for parent in self._parent_citables():
            if parent is None or id(parent) in _visited:
                continue
            for label, key, val in parent.effective_citations(_visited=_visited):
                if key not in seen_keys:
                    result.append((label, key, val))
                    seen_keys.add(key)

        return result
