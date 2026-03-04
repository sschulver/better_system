"""
Core data models for the ontological classification system.

Node classification format: object_class:domain:category:type:name

Examples:
  concept:demography:*:domain:demography
  concept:demography:population-distribution:category:population-distribution
  observable:demography:population-distribution:spatial-aggregate:population-spatial-aggregate
"""
from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeClassification:
    """
    Parses and holds the five-part classification string:
    object_class:domain:category:type:name

    Use '*' for category when the node is a domain-level concept.
    """
    object_class: str
    domain: str
    category: str
    type: str
    name: str

    @property
    def classification_str(self) -> str:
        return f"{self.object_class}:{self.domain}:{self.category}:{self.type}:{self.name}"

    @classmethod
    def parse(cls, s: str) -> NodeClassification:
        parts = s.split(":", 4)
        if len(parts) != 5:
            raise ValueError(
                f"Invalid classification '{s}'. "
                "Expected: object_class:domain:category:type:name"
            )
        return cls(
            object_class=parts[0],
            domain=parts[1],
            category=parts[2],
            type=parts[3],
            name=parts[4],
        )

    def __str__(self) -> str:
        return self.classification_str


@dataclass
class FieldDefinition:
    """
    A canonical field name associated with an observable.

    Captures the "ideal" name for the field and any common aliases so that
    incoming dataset columns can be matched against it.
    """
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aliases: list[str] = field(default_factory=list)
    data_type: Optional[str] = None   # e.g. "integer", "float", "string"
    description: Optional[str] = None

    def all_names(self) -> list[str]:
        """Return the canonical name plus all aliases."""
        return [self.name] + list(self.aliases)


@dataclass
class DatasetClassification:
    """
    Links a known dataset to an observable, recording which field names in
    that dataset correspond to the observable.

    Multiple DatasetClassifications can be attached to a single observable
    (e.g. the same concept appears in both a census and a survey).
    """
    dataset_name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None          # e.g. "US Census Bureau"
    dataset_type: Optional[str] = None    # e.g. "census", "survey", "administrative"
    description: Optional[str] = None
    # Actual column/field names as they appear in this specific dataset
    field_names: list[str] = field(default_factory=list)
    # Alternative names this dataset is known by (for fuzzy matching)
    aliases: list[str] = field(default_factory=list)

    def all_dataset_names(self) -> list[str]:
        """Return the primary dataset name plus all aliases."""
        return [self.dataset_name] + list(self.aliases)


@dataclass
class OntologyNode:
    """
    A node in the ontological hierarchy.

    Hierarchy levels (determined by classification.type):
      - Domain nodes:   type == "domain",   object_class == "concept"
      - Category nodes: type == "category", object_class == "concept"
      - Observables:    object_class == "observable", type is a specific observable kind

    Only observable nodes carry field_definitions and dataset_classifications.
    """
    classification: NodeClassification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    field_definitions: list[FieldDefinition] = field(default_factory=list)
    dataset_classifications: list[DatasetClassification] = field(default_factory=list)

    # --- convenience properties ---

    @property
    def is_domain(self) -> bool:
        return self.classification.type == "domain"

    @property
    def is_category(self) -> bool:
        return self.classification.type == "category"

    @property
    def is_observable(self) -> bool:
        return self.classification.object_class == "observable"

    # --- field helpers ---

    def all_field_names(self) -> list[str]:
        """
        All field names associated with this observable: canonical names,
        aliases, and dataset-specific column names.
        """
        names: set[str] = set()
        for fd in self.field_definitions:
            names.update(fd.all_names())
        for dc in self.dataset_classifications:
            names.update(dc.field_names)
        return list(names)

    def add_field_definition(self, fd: FieldDefinition) -> None:
        self.field_definitions.append(fd)

    def add_dataset_classification(self, dc: DatasetClassification) -> None:
        self.dataset_classifications.append(dc)


@dataclass
class ConfirmedMapping:
    """
    A persisted record that a specific dataset field was confirmed to map to
    a specific observable node (and optionally a canonical FieldDefinition).

    Created either automatically (source="auto") when a caller accepts a
    DatasetSearchResult, or manually (source="manual") by a user.

    Once confirmed, the mapping serves as institutional memory: future searches
    can surface prior confirmations and the dataset field can be promoted into
    the node's DatasetClassification so it improves future matching.
    """
    node_id: str
    dataset_name: str
    dataset_field: str               # incoming column name
    canonical_field: Optional[str]   # FieldDefinition.name, or None if new/unknown
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: Optional[float] = None  # similarity score; None for manual entries
    source: str = "manual"              # "auto" | "manual"
    notes: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
