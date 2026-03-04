#!/usr/bin/env python3
"""
Command-line interface for the ontological classification system.

Usage examples
--------------
# Add a domain
python cli.py add-node concept:demography:*:domain:demography \
    --description "Demographic domain"

# Add a category
python cli.py add-node \
    concept:demography:population-distribution:category:population-distribution

# Add an observable with fields and a known dataset
python cli.py add-node \
    observable:demography:population-distribution:spatial-aggregate:population-spatial-aggregate \
    --description "Spatially aggregated population counts"

python cli.py add-field \\
    observable:demography:population-distribution:spatial-aggregate:population-spatial-aggregate \\
    total_population --alias pop_total --alias population --type integer

python cli.py add-dataset \\
    observable:demography:population-distribution:spatial-aggregate:population-spatial-aggregate \\
    "American Community Survey" --source "US Census Bureau" --type survey \\
    --field B01003_001E --field total_pop --alias ACS --alias ACS5

# Search for matching observables for an incoming dataset
python cli.py search --dataset "ACS" --field total_pop --field pop_density

# Show the full ontology tree
python cli.py tree

# List observables in a domain
python cli.py list --domain demography --type observable
"""
from __future__ import annotations

import sys
from typing import Optional

import click

from ontology import (
    DatasetClassification,
    FieldDefinition,
    NodeClassification,
    OntologyNode,
    OntologySearcher,
    OntologyStore,
)
from ontology.models import ConfirmedMapping

DEFAULT_DB = "ontology.db"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--db",
    default=DEFAULT_DB,
    show_default=True,
    envvar="ONTOLOGY_DB",
    help="Path to the SQLite database file.",
)
@click.pass_context
def cli(ctx: click.Context, db: str) -> None:
    """Ontological classification system."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


def _store(ctx: click.Context) -> OntologyStore:
    return OntologyStore(ctx.obj["db"])


# ---------------------------------------------------------------------------
# add-node
# ---------------------------------------------------------------------------

@cli.command("add-node")
@click.argument("classification")
@click.option("--description", "-d", default=None, help="Human-readable description.")
@click.pass_context
def add_node(ctx: click.Context, classification: str, description: Optional[str]) -> None:
    """Add or update an ontology node.

    CLASSIFICATION is the full five-part string:
    object_class:domain:category:type:name

    \b
    Examples:
      concept:demography:*:domain:demography
      concept:demography:population-distribution:category:population-distribution
      observable:demography:population-distribution:spatial-aggregate:pop-spatial-aggregate
    """
    try:
        nc = NodeClassification.parse(classification)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    store = _store(ctx)
    existing = store.get_node_by_classification(classification)

    if existing:
        if description:
            store.update_description(existing.id, description)
            click.echo(f"Updated description for {classification}")
        else:
            click.echo(f"Node already exists: {classification}")
    else:
        node = OntologyNode(classification=nc, description=description)
        store.add_node(node)
        click.echo(f"Added: {classification}")

    store.close()


# ---------------------------------------------------------------------------
# add-field
# ---------------------------------------------------------------------------

@cli.command("add-field")
@click.argument("classification")
@click.argument("field_name")
@click.option("--alias", "-a", multiple=True, help="Alias for this field (repeatable).")
@click.option("--type", "data_type", default=None, help="Data type (e.g. integer, float, string).")
@click.option("--description", "-d", default=None)
@click.pass_context
def add_field(
    ctx: click.Context,
    classification: str,
    field_name: str,
    alias: tuple[str, ...],
    data_type: Optional[str],
    description: Optional[str],
) -> None:
    """Add a canonical field definition to an observable.

    CLASSIFICATION is the full five-part node classification string.
    FIELD_NAME is the canonical name for this field.
    """
    store = _store(ctx)
    node = store.get_node_by_classification(classification)
    if not node:
        click.echo(f"Node not found: {classification}", err=True)
        sys.exit(1)

    fd = FieldDefinition(
        name=field_name,
        aliases=list(alias),
        data_type=data_type,
        description=description,
    )
    store.add_field_definition(node.id, fd)
    click.echo(
        f"Added field '{field_name}'"
        + (f" (aliases: {', '.join(alias)})" if alias else "")
        + f" to {classification}"
    )
    store.close()


# ---------------------------------------------------------------------------
# add-dataset
# ---------------------------------------------------------------------------

@cli.command("add-dataset")
@click.argument("classification")
@click.argument("dataset_name")
@click.option("--source", "-s", default=None, help="Data source / publisher.")
@click.option(
    "--type", "dataset_type", default=None,
    help="Dataset type (e.g. census, survey, administrative).",
)
@click.option("--description", "-d", default=None)
@click.option("--field", "-f", multiple=True, help="Field/column name in this dataset (repeatable).")
@click.option("--alias", "-a", multiple=True, help="Alternative dataset name (repeatable).")
@click.pass_context
def add_dataset(
    ctx: click.Context,
    classification: str,
    dataset_name: str,
    source: Optional[str],
    dataset_type: Optional[str],
    description: Optional[str],
    field: tuple[str, ...],
    alias: tuple[str, ...],
) -> None:
    """Attach a known dataset classification to an observable.

    CLASSIFICATION is the full five-part node classification string.
    DATASET_NAME is the primary name for this dataset.
    """
    store = _store(ctx)
    node = store.get_node_by_classification(classification)
    if not node:
        click.echo(f"Node not found: {classification}", err=True)
        sys.exit(1)

    dc = DatasetClassification(
        dataset_name=dataset_name,
        source=source,
        dataset_type=dataset_type,
        description=description,
        field_names=list(field),
        aliases=list(alias),
    )
    store.add_dataset_classification(node.id, dc)
    lines = [f"Added dataset '{dataset_name}' to {classification}"]
    if field:
        lines.append(f"  fields:  {', '.join(field)}")
    if alias:
        lines.append(f"  aliases: {', '.join(alias)}")
    click.echo("\n".join(lines))
    store.close()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command("search")
@click.option("--dataset", "-D", default=None, help="Incoming dataset name.")
@click.option("--field", "-f", multiple=True, help="Incoming field/column name (repeatable).")
@click.option("--domain", default=None, help="Restrict search to this domain.")
@click.option("--category", default=None, help="Restrict search to this category.")
@click.option("--threshold", default=0.55, show_default=True, help="Minimum similarity score.")
@click.option("--top", default=10, show_default=True, help="Number of results to return.")
@click.pass_context
def search(
    ctx: click.Context,
    dataset: Optional[str],
    field: tuple[str, ...],
    domain: Optional[str],
    category: Optional[str],
    threshold: float,
    top: int,
) -> None:
    """Search observables for an incoming dataset.

    Provide --dataset and/or one or more --field options.

    \b
    Example:
      python cli.py search --dataset "ACS 5-year" --field total_pop --field B01003_001E
    """
    if not dataset and not field:
        click.echo("Provide at least --dataset or one --field option.", err=True)
        sys.exit(1)

    store = _store(ctx)
    searcher = OntologySearcher(store, threshold=threshold)

    results = searcher.search(
        dataset_name=dataset or None,
        field_names=list(field) or None,
        domain=domain,
        category=category,
        top_k=top,
    )

    if not results:
        click.echo("No matches found.")
    else:
        click.echo(f"Top {len(results)} match(es):\n")
        for i, r in enumerate(results, 1):
            click.echo(f"[{i}]")
            click.echo(r.summary())
            click.echo()

    store.close()


# ---------------------------------------------------------------------------
# suggest
# ---------------------------------------------------------------------------

@cli.command("suggest")
@click.option("--field", "-f", multiple=True, required=True, help="Incoming field name (repeatable).")
@click.option("--domain", default=None)
@click.option("--category", default=None)
@click.option("--threshold", default=0.55, show_default=True)
@click.option("--top", default=3, show_default=True)
@click.pass_context
def suggest(
    ctx: click.Context,
    field: tuple[str, ...],
    domain: Optional[str],
    category: Optional[str],
    threshold: float,
    top: int,
) -> None:
    """Suggest observable classifications per incoming field name.

    Unlike 'search', this ranks observables independently for each field,
    useful when auto-populating a schema mapping.

    \b
    Example:
      python cli.py suggest --field total_pop --field pop_density --field geoid
    """
    store = _store(ctx)
    searcher = OntologySearcher(store, threshold=threshold)

    suggestions = searcher.suggest_field_mappings(
        field_names=list(field),
        domain=domain,
        category=category,
        top_k=top,
    )

    for fname, results in suggestions.items():
        click.echo(f"Field '{fname}':")
        if not results:
            click.echo("  (no matches)")
        else:
            for r in results:
                fm = r.field_matches[0]
                canonical = r.suggested_mapping.get(fname, fm.matched_field)
                click.echo(
                    f"  [{fm.score:.2f}] {r.classification}"
                    f"  ({fname!r} -> canonical: {canonical!r})"
                )
        click.echo()

    store.close()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.option(
    "--type", "node_type",
    type=click.Choice(["domain", "category", "observable", "all"], case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option("--domain", default=None)
@click.option("--category", default=None)
@click.pass_context
def list_nodes(
    ctx: click.Context,
    node_type: str,
    domain: Optional[str],
    category: Optional[str],
) -> None:
    """List nodes in the ontology."""
    store = _store(ctx)

    if node_type == "domain":
        nodes = store.list_domains()
    elif node_type == "category":
        nodes = store.list_categories(domain=domain)
    elif node_type == "observable":
        nodes = store.list_observables(domain=domain, category=category)
    else:
        nodes = store.list_all()

    if not nodes:
        click.echo("(empty)")
    else:
        for n in nodes:
            desc = f"  # {n.description}" if n.description else ""
            fields_info = ""
            if n.field_definitions:
                fields_info = f"  [{len(n.field_definitions)} field(s)]"
            datasets_info = ""
            if n.dataset_classifications:
                datasets_info = f"  [{len(n.dataset_classifications)} dataset(s)]"
            click.echo(
                f"{n.classification.classification_str}"
                f"{fields_info}{datasets_info}{desc}"
            )

    store.close()


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

@cli.command("tree")
@click.option("--domain", default=None, help="Show only this domain.")
@click.pass_context
def tree(ctx: click.Context, domain: Optional[str]) -> None:
    """Display the ontology as a hierarchy tree."""
    store = _store(ctx)

    domains = store.list_domains()
    if domain:
        domains = [d for d in domains if d.classification.domain == domain]

    if not domains:
        click.echo("(no domains found)")
        store.close()
        return

    for dom in domains:
        dn = dom.classification.name
        click.echo(f"{dn}/")
        categories = store.list_categories(domain=dn)
        for cat in categories:
            cn = cat.classification.category
            click.echo(f"  {cn}/")
            observables = store.list_observables(domain=dn, category=cn)
            for obs in observables:
                label = obs.classification.name
                fd_count = len(obs.field_definitions)
                dc_count = len(obs.dataset_classifications)
                click.echo(
                    f"    {label}"
                    + (f"  [{fd_count}F]" if fd_count else "")
                    + (f"  [{dc_count}D]" if dc_count else "")
                )

    store.close()


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

@cli.command("show")
@click.argument("classification")
@click.pass_context
def show(ctx: click.Context, classification: str) -> None:
    """Show full details of a single node.

    CLASSIFICATION is the full five-part classification string.
    """
    store = _store(ctx)
    node = store.get_node_by_classification(classification)
    if not node:
        click.echo(f"Not found: {classification}", err=True)
        sys.exit(1)

    nc = node.classification
    click.echo(f"Classification : {nc.classification_str}")
    click.echo(f"ID             : {node.id}")
    if node.description:
        click.echo(f"Description    : {node.description}")

    if node.field_definitions:
        click.echo("\nField Definitions:")
        for fd in node.field_definitions:
            line = f"  {fd.name}"
            if fd.data_type:
                line += f" ({fd.data_type})"
            if fd.aliases:
                line += f"  aliases: {', '.join(fd.aliases)}"
            if fd.description:
                line += f"  — {fd.description}"
            click.echo(line)

    if node.dataset_classifications:
        click.echo("\nDataset Classifications:")
        for dc in node.dataset_classifications:
            click.echo(f"  {dc.dataset_name}")
            if dc.source:
                click.echo(f"    source      : {dc.source}")
            if dc.dataset_type:
                click.echo(f"    type        : {dc.dataset_type}")
            if dc.description:
                click.echo(f"    description : {dc.description}")
            if dc.field_names:
                click.echo(f"    fields      : {', '.join(dc.field_names)}")
            if dc.aliases:
                click.echo(f"    aliases     : {', '.join(dc.aliases)}")

    store.close()


# ---------------------------------------------------------------------------
# confirm-mapping
# ---------------------------------------------------------------------------

@cli.command("confirm-mapping")
@click.argument("classification")
@click.argument("dataset_name")
@click.argument("dataset_field")
@click.option(
    "--canonical", "-c", default=None,
    help="Canonical FieldDefinition name this field maps to.",
)
@click.option(
    "--confidence", default=None, type=float,
    help="Similarity confidence score (0.0–1.0).",
)
@click.option(
    "--source", default="manual",
    type=click.Choice(["manual", "auto"], case_sensitive=False),
    show_default=True,
)
@click.option("--notes", "-n", default=None, help="Optional notes.")
@click.pass_context
def confirm_mapping(
    ctx: click.Context,
    classification: str,
    dataset_name: str,
    dataset_field: str,
    canonical: Optional[str],
    confidence: Optional[float],
    source: str,
    notes: Optional[str],
) -> None:
    """Record that DATASET_FIELD from DATASET_NAME maps to a node.

    CLASSIFICATION is the full five-part node classification string.

    \b
    Example:
      python cli.py confirm-mapping \\
        observable:demography:population-distribution:spatial-aggregate:population-spatial-aggregate \\
        "My Census 2024" total_pop --canonical total_population --confidence 0.92
    """
    store = _store(ctx)
    node = store.get_node_by_classification(classification)
    if not node:
        click.echo(f"Node not found: {classification}", err=True)
        sys.exit(1)

    mapping = store.confirm_mapping(
        node_id=node.id,
        dataset_name=dataset_name,
        dataset_field=dataset_field,
        canonical_field=canonical,
        confidence=confidence,
        source=source,
        notes=notes,
    )
    click.echo(
        f"Confirmed: '{dataset_field}' from '{dataset_name}'"
        f" -> {classification}"
        + (f" (canonical: {canonical})" if canonical else "")
        + (f"  [{confidence:.2f}]" if confidence is not None else "")
    )
    store.close()


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

@cli.command("history")
@click.option("--dataset", "-D", default=None, help="Filter by dataset name.")
@click.option("--field", "-f", default=None, help="Filter by dataset field name.")
@click.option("--classification", default=None, help="Filter by node classification.")
@click.option(
    "--source", default=None,
    type=click.Choice(["manual", "auto"], case_sensitive=False),
)
@click.pass_context
def history(
    ctx: click.Context,
    dataset: Optional[str],
    field: Optional[str],
    classification: Optional[str],
    source: Optional[str],
) -> None:
    """Show confirmed field mapping history.

    At least one of --dataset, --field, or --classification should be provided
    to avoid listing every mapping in the database.

    \b
    Examples:
      python cli.py history --dataset "My Census 2024"
      python cli.py history --dataset "My Census 2024" --field total_pop
      python cli.py history --classification observable:demography:...:population-spatial-aggregate
    """
    store = _store(ctx)

    node_id: Optional[str] = None
    if classification:
        node = store.get_node_by_classification(classification)
        if not node:
            click.echo(f"Node not found: {classification}", err=True)
            sys.exit(1)
        node_id = node.id

    if dataset and field:
        mappings = store.get_field_history(dataset_name=dataset, dataset_field=field)
    elif dataset:
        mappings = store.get_field_history(dataset_name=dataset)
    else:
        mappings = store.get_confirmed_mappings(node_id=node_id, source=source)

    if not mappings:
        click.echo("(no confirmed mappings found)")
    else:
        for m in mappings:
            node_label = m.node_id
            # Try to resolve node classification for readability
            node_obj = store.get_node(m.node_id)
            if node_obj:
                node_label = node_obj.classification.classification_str

            conf_str = f"  [{m.confidence:.2f}]" if m.confidence is not None else ""
            canonical_str = f"  -> {m.canonical_field}" if m.canonical_field else ""
            click.echo(
                f"{m.created_at[:19]}  [{m.source}]{conf_str}"
                f"  '{m.dataset_field}'{canonical_str}"
                f"  @ {node_label}"
            )
            if m.notes:
                click.echo(f"    notes: {m.notes}")

    store.close()


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

@cli.command("delete")
@click.argument("classification")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def delete(ctx: click.Context, classification: str, yes: bool) -> None:
    """Delete a node from the ontology (cascades to fields and datasets)."""
    store = _store(ctx)
    node = store.get_node_by_classification(classification)
    if not node:
        click.echo(f"Not found: {classification}", err=True)
        sys.exit(1)

    if not yes:
        click.confirm(f"Delete '{classification}'?", abort=True)

    store.delete_node(node.id)
    click.echo(f"Deleted: {classification}")
    store.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
