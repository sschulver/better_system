"""
SQLite-backed storage for the ontological classification system.

Schema overview
---------------
nodes                   – one row per OntologyNode
field_definitions       – canonical fields attached to a node
field_aliases           – aliases for a field definition
dataset_classifications – known datasets attached to a node
dataset_field_names     – actual column names within a dataset
dataset_aliases         – alternative names a dataset is known by
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .models import (
    DatasetClassification,
    FieldDefinition,
    NodeClassification,
    OntologyNode,
)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    object_class TEXT NOT NULL,
    domain      TEXT NOT NULL,
    category    TEXT NOT NULL,
    node_type   TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_domain        ON nodes (domain);
CREATE INDEX IF NOT EXISTS idx_nodes_object_class  ON nodes (object_class);
CREATE INDEX IF NOT EXISTS idx_nodes_node_type     ON nodes (node_type);
CREATE UNIQUE INDEX IF NOT EXISTS uidx_nodes_classification
    ON nodes (object_class, domain, category, node_type, name);

CREATE TABLE IF NOT EXISTS field_definitions (
    id          TEXT PRIMARY KEY,
    node_id     TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    data_type   TEXT,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_fd_node_id ON field_definitions (node_id);

CREATE TABLE IF NOT EXISTS field_aliases (
    field_id TEXT NOT NULL REFERENCES field_definitions (id) ON DELETE CASCADE,
    alias    TEXT NOT NULL,
    PRIMARY KEY (field_id, alias)
);

CREATE TABLE IF NOT EXISTS dataset_classifications (
    id           TEXT PRIMARY KEY,
    node_id      TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    dataset_name TEXT NOT NULL,
    source       TEXT,
    dataset_type TEXT,
    description  TEXT
);

CREATE INDEX IF NOT EXISTS idx_dc_node_id      ON dataset_classifications (node_id);
CREATE INDEX IF NOT EXISTS idx_dc_dataset_name ON dataset_classifications (dataset_name);

CREATE TABLE IF NOT EXISTS dataset_field_names (
    dataset_id TEXT NOT NULL REFERENCES dataset_classifications (id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    PRIMARY KEY (dataset_id, field_name)
);

CREATE TABLE IF NOT EXISTS dataset_aliases (
    dataset_id TEXT NOT NULL REFERENCES dataset_classifications (id) ON DELETE CASCADE,
    alias      TEXT NOT NULL,
    PRIMARY KEY (dataset_id, alias)
);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class OntologyStore:
    """
    Persistent SQLite store for OntologyNode objects.

    Usage::

        store = OntologyStore("ontology.db")
        store.add_node(node)
        node = store.get_node_by_classification("observable:demography:...")
        store.close()

    Pass ``":memory:"`` for an in-memory database (useful for tests).
    """

    def __init__(self, db_path: str | Path = "ontology.db") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def add_node(self, node: OntologyNode) -> OntologyNode:
        """
        Insert or replace a node and all its child objects.

        If a node with the same classification already exists it is fully
        replaced (including its field definitions and dataset classifications).
        """
        with self._conn:
            c = self._conn.cursor()

            # Upsert the node row
            c.execute(
                """
                INSERT INTO nodes (id, object_class, domain, category, node_type, name, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (object_class, domain, category, node_type, name)
                DO UPDATE SET
                    id          = excluded.id,
                    description = excluded.description
                """,
                (
                    node.id,
                    node.classification.object_class,
                    node.classification.domain,
                    node.classification.category,
                    node.classification.type,
                    node.classification.name,
                    node.description,
                ),
            )

            # Delete stale child rows (cascade would handle DELETE but we also
            # do INSERT OR REPLACE so clear manually for a full replace).
            c.execute("DELETE FROM field_definitions WHERE node_id = ?", (node.id,))
            c.execute(
                "DELETE FROM dataset_classifications WHERE node_id = ?", (node.id,)
            )

            self._insert_field_definitions(c, node.id, node.field_definitions)
            self._insert_dataset_classifications(
                c, node.id, node.dataset_classifications
            )

        return node

    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Fetch a node by its UUID."""
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_node_by_classification(self, classification: str) -> Optional[OntologyNode]:
        """Fetch a node by its full classification string."""
        nc = NodeClassification.parse(classification)
        row = self._conn.execute(
            """
            SELECT * FROM nodes
            WHERE object_class = ? AND domain = ? AND category = ?
              AND node_type = ? AND name = ?
            """,
            (nc.object_class, nc.domain, nc.category, nc.type, nc.name),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def update_description(self, node_id: str, description: str) -> bool:
        """Update the description of an existing node."""
        with self._conn:
            result = self._conn.execute(
                "UPDATE nodes SET description = ? WHERE id = ?",
                (description, node_id),
            )
        return result.rowcount > 0

    def delete_node(self, node_id: str) -> bool:
        """Delete a node (cascades to child rows)."""
        with self._conn:
            result = self._conn.execute(
                "DELETE FROM nodes WHERE id = ?", (node_id,)
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Field definition helpers
    # ------------------------------------------------------------------

    def add_field_definition(self, node_id: str, fd: FieldDefinition) -> None:
        """Append a field definition to an existing node."""
        with self._conn:
            c = self._conn.cursor()
            self._insert_field_definitions(c, node_id, [fd])

    def remove_field_definition(self, field_id: str) -> bool:
        with self._conn:
            result = self._conn.execute(
                "DELETE FROM field_definitions WHERE id = ?", (field_id,)
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Dataset classification helpers
    # ------------------------------------------------------------------

    def add_dataset_classification(
        self, node_id: str, dc: DatasetClassification
    ) -> None:
        """Append a dataset classification to an existing node."""
        with self._conn:
            c = self._conn.cursor()
            self._insert_dataset_classifications(c, node_id, [dc])

    def remove_dataset_classification(self, dataset_id: str) -> bool:
        with self._conn:
            result = self._conn.execute(
                "DELETE FROM dataset_classifications WHERE id = ?", (dataset_id,)
            )
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Listing / querying
    # ------------------------------------------------------------------

    def list_domains(self) -> list[OntologyNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE node_type = 'domain' ORDER BY domain"
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def list_categories(self, domain: Optional[str] = None) -> list[OntologyNode]:
        if domain:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE node_type = 'category' AND domain = ?"
                " ORDER BY name",
                (domain,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE node_type = 'category'"
                " ORDER BY domain, name"
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def list_observables(
        self,
        domain: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[OntologyNode]:
        query = "SELECT * FROM nodes WHERE object_class = 'observable'"
        params: list[str] = []
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY domain, category, name"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_node(r) for r in rows]

    def list_all(self) -> list[OntologyNode]:
        rows = self._conn.execute(
            "SELECT * FROM nodes ORDER BY domain, category, node_type, name"
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def search_nodes(self, text: str) -> list[OntologyNode]:
        """
        Simple substring search across domain, category, node_type, name, and
        description. For richer fuzzy matching use OntologySearcher.
        """
        like = f"%{text.lower()}%"
        rows = self._conn.execute(
            """
            SELECT * FROM nodes
            WHERE lower(domain)      LIKE ?
               OR lower(category)    LIKE ?
               OR lower(node_type)   LIKE ?
               OR lower(name)        LIKE ?
               OR lower(coalesce(description, '')) LIKE ?
            ORDER BY domain, category, name
            """,
            (like, like, like, like, like),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _insert_field_definitions(
        self, cursor: sqlite3.Cursor, node_id: str, fds: list[FieldDefinition]
    ) -> None:
        for fd in fds:
            cursor.execute(
                """
                INSERT OR REPLACE INTO field_definitions
                    (id, node_id, name, data_type, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                (fd.id, node_id, fd.name, fd.data_type, fd.description),
            )
            for alias in fd.aliases:
                cursor.execute(
                    "INSERT OR IGNORE INTO field_aliases (field_id, alias)"
                    " VALUES (?, ?)",
                    (fd.id, alias),
                )

    def _insert_dataset_classifications(
        self,
        cursor: sqlite3.Cursor,
        node_id: str,
        dcs: list[DatasetClassification],
    ) -> None:
        for dc in dcs:
            cursor.execute(
                """
                INSERT OR REPLACE INTO dataset_classifications
                    (id, node_id, dataset_name, source, dataset_type, description)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dc.id,
                    node_id,
                    dc.dataset_name,
                    dc.source,
                    dc.dataset_type,
                    dc.description,
                ),
            )
            for fn in dc.field_names:
                cursor.execute(
                    "INSERT OR IGNORE INTO dataset_field_names"
                    " (dataset_id, field_name) VALUES (?, ?)",
                    (dc.id, fn),
                )
            for alias in dc.aliases:
                cursor.execute(
                    "INSERT OR IGNORE INTO dataset_aliases (dataset_id, alias)"
                    " VALUES (?, ?)",
                    (dc.id, alias),
                )

    def _row_to_node(self, row: sqlite3.Row) -> OntologyNode:
        node_id = row["id"]

        # Field definitions
        fd_rows = self._conn.execute(
            "SELECT * FROM field_definitions WHERE node_id = ?", (node_id,)
        ).fetchall()
        field_defs: list[FieldDefinition] = []
        for fd_row in fd_rows:
            alias_rows = self._conn.execute(
                "SELECT alias FROM field_aliases WHERE field_id = ?", (fd_row["id"],)
            ).fetchall()
            field_defs.append(
                FieldDefinition(
                    id=fd_row["id"],
                    name=fd_row["name"],
                    data_type=fd_row["data_type"],
                    description=fd_row["description"],
                    aliases=[r["alias"] for r in alias_rows],
                )
            )

        # Dataset classifications
        dc_rows = self._conn.execute(
            "SELECT * FROM dataset_classifications WHERE node_id = ?", (node_id,)
        ).fetchall()
        dataset_classes: list[DatasetClassification] = []
        for dc_row in dc_rows:
            fn_rows = self._conn.execute(
                "SELECT field_name FROM dataset_field_names WHERE dataset_id = ?",
                (dc_row["id"],),
            ).fetchall()
            alias_rows = self._conn.execute(
                "SELECT alias FROM dataset_aliases WHERE dataset_id = ?",
                (dc_row["id"],),
            ).fetchall()
            dataset_classes.append(
                DatasetClassification(
                    id=dc_row["id"],
                    dataset_name=dc_row["dataset_name"],
                    source=dc_row["source"],
                    dataset_type=dc_row["dataset_type"],
                    description=dc_row["description"],
                    field_names=[r["field_name"] for r in fn_rows],
                    aliases=[r["alias"] for r in alias_rows],
                )
            )

        return OntologyNode(
            id=node_id,
            classification=NodeClassification(
                object_class=row["object_class"],
                domain=row["domain"],
                category=row["category"],
                type=row["node_type"],
                name=row["name"],
            ),
            description=row["description"],
            field_definitions=field_defs,
            dataset_classifications=dataset_classes,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> OntologyStore:
        return self

    def __exit__(self, *_) -> None:
        self.close()
