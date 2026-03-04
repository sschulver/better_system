"""
Ontological classification system.

Quick start::

    from ontology import OntologyStore, OntologySearcher
    from ontology.models import (
        NodeClassification,
        OntologyNode,
        FieldDefinition,
        DatasetClassification,
    )

    store = OntologyStore("ontology.db")
    searcher = OntologySearcher(store)

    results = searcher.search(
        dataset_name="American Community Survey",
        field_names=["total_pop", "pop_density"],
    )
    for r in results:
        print(r.summary())
"""
from .models import (
    ConfirmedMapping,
    DatasetClassification,
    FieldDefinition,
    NodeClassification,
    OntologyNode,
)
from .search import OntologySearcher, DatasetSearchResult
from .store import OntologyStore

__all__ = [
    "ConfirmedMapping",
    "DatasetClassification",
    "DatasetSearchResult",
    "FieldDefinition",
    "NodeClassification",
    "OntologyNode",
    "OntologySearcher",
    "OntologyStore",
]
