#!/usr/bin/env python3
"""
Interactive TUI for the ontological classification system.

Layout
------
┌─────────────────────────────────────────────────────────┐
│ [Browse]  [Search Dataset]                               │
├──────────────────────┬──────────────────────────────────┤
│ 📁 demography        │ observable:demography:…           │
│   📂 pop-dist        │                                   │
│     🔵 pop-spatial   │ Field Definitions                 │
│       …              │   • total_population (integer)    │
│                      │     aliases: pop_total, …         │
│                      │                                   │
│                      │ Dataset Classifications            │
│                      │   • American Community Survey     │
│                      │     fields: B01003_001E, …        │
├──────────────────────┤                                   │
│ 🔍 filter…           │ Confirmed Mappings (3)            │
│                      │   • 2024-01-01 [auto] …           │
└──────────────────────┴──────────────────────────────────┘

Keybindings
-----------
  q / Ctrl+C  quit
  r           refresh tree from DB
  f  or  /    jump to tree filter input
  Tab         cycle focus
  ↑ ↓ ← →    navigate tree
  Space/Enter expand/collapse tree node
"""
from __future__ import annotations

from typing import Optional

import click
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)
from textual import on

from ontology import OntologyNode, OntologySearcher, OntologyStore


# ---------------------------------------------------------------------------
# Tree node metadata (stored in Tree node .data)
# ---------------------------------------------------------------------------

class _DomainMeta:
    __slots__ = ("domain",)

    def __init__(self, domain: str) -> None:
        self.domain = domain


class _CategoryMeta:
    __slots__ = ("domain", "category")

    def __init__(self, domain: str, category: str) -> None:
        self.domain = domain
        self.category = category


class _ObservableMeta:
    __slots__ = ("node",)

    def __init__(self, node: OntologyNode) -> None:
        self.node = node


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class OntologyBrowser(App):
    """Interactive browser for the ontological classification system."""

    CSS = """
    /* ── global ── */
    Screen {
        layers: base;
    }

    /* ── browse tab ── */
    #browse-pane {
        layout: horizontal;
        height: 1fr;
    }

    #tree-side {
        width: 1fr;
        min-width: 22;
        max-width: 40;
        border-right: solid $panel-darken-1;
    }

    #tree-scroll {
        height: 1fr;
    }

    #filter-input {
        height: 3;
        border-top: solid $panel-darken-1;
    }

    #detail-scroll {
        width: 2fr;
        padding: 1 2;
    }

    /* ── search tab ── */
    #search-pane {
        layout: vertical;
        padding: 1 2;
        height: 1fr;
    }

    #search-form {
        height: auto;
        max-height: 15;
        border: round $primary;
        padding: 1 2;
        margin-bottom: 1;
    }

    #fields-input {
        height: 5;
    }

    #form-controls {
        height: 3;
        margin-top: 1;
        align: left middle;
    }

    .form-label {
        height: 1;
        margin-top: 1;
        color: $text-disabled;
    }

    .inline-label {
        height: 3;
        content-align: left middle;
        padding: 0 1;
    }

    .narrow-input {
        width: 14;
    }

    #results-scroll {
        height: 1fr;
        border: round $panel;
        padding: 1 2;
    }

    #confirm-bar {
        height: 3;
        align: left middle;
    }

    #confirm-status {
        padding: 0 2;
        content-align: left middle;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "focus_filter", "Filter"),
        Binding("/", "focus_filter", "Filter", show=False),
    ]

    def __init__(self, db_path: str = "ontology.db") -> None:
        super().__init__()
        self.db_path = db_path
        self.store = OntologyStore(db_path)
        self.searcher = OntologySearcher(self.store)
        self._last_results: list = []

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent(initial="browse"):
            # ── Browse ──────────────────────────────────────────────
            with TabPane("Browse", id="browse"):
                with Horizontal(id="browse-pane"):
                    # left: tree + filter
                    with Vertical(id="tree-side"):
                        with ScrollableContainer(id="tree-scroll"):
                            yield Tree("Ontology", id="ontology-tree")
                        yield Input(placeholder="🔍 filter…", id="filter-input")
                    # right: detail
                    with ScrollableContainer(id="detail-scroll"):
                        yield Static(
                            "Select a node from the tree to view details.\n\n"
                            "[dim]Arrow keys navigate · Space/Enter expands · / filters[/dim]",
                            id="detail-content",
                        )

            # ── Search Dataset ───────────────────────────────────────
            with TabPane("Search Dataset", id="search"):
                with Vertical(id="search-pane"):
                    with Vertical(id="search-form"):
                        yield Label("Dataset name:", classes="form-label")
                        yield Input(
                            placeholder="e.g. American Community Survey",
                            id="ds-name-input",
                        )
                        yield Label(
                            "Field names — one per line:",
                            classes="form-label",
                        )
                        yield TextArea(id="fields-input")
                        with Horizontal(id="form-controls"):
                            yield Button("Search", id="search-btn", variant="primary")
                            yield Label("Domain:", classes="inline-label")
                            yield Input(
                                placeholder="all",
                                id="domain-filter",
                                classes="narrow-input",
                            )
                            yield Label("Threshold:", classes="inline-label")
                            yield Input(
                                "0.55",
                                id="threshold-input",
                                classes="narrow-input",
                            )

                    with ScrollableContainer(id="results-scroll"):
                        yield Static(
                            "Enter a dataset name and/or field names, then press [bold]Search[/bold].",
                            id="results-content",
                        )

                    with Horizontal(id="confirm-bar"):
                        yield Button(
                            "Confirm Top",
                            id="confirm-top-btn",
                            variant="success",
                            disabled=True,
                        )
                        yield Button(
                            "Confirm All",
                            id="confirm-all-btn",
                            variant="success",
                            disabled=True,
                        )
                        yield Static("", id="confirm-status")

        yield Footer()

    def on_mount(self) -> None:
        self._rebuild_tree()

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def _rebuild_tree(self, filter_text: str = "") -> None:
        """Rebuild the hierarchy tree, optionally filtered by text."""
        tree = self.query_one("#ontology-tree", Tree)
        tree.clear()
        tree.root.expand()

        fl = filter_text.lower().strip()

        for dom_node in self.store.list_domains():
            domain = dom_node.classification.domain
            cats = self.store.list_categories(domain=domain)
            obs_all = self.store.list_observables(domain=domain)

            # Skip domain entirely if filter matches nothing inside it
            if fl and not self._domain_matches(fl, domain, cats, obs_all):
                continue

            expand_domain = bool(fl)
            n_dom = tree.root.add(
                f"📁 {domain}",
                data=_DomainMeta(domain),
                expand=expand_domain,
            )

            for cat_node in cats:
                category = cat_node.classification.category
                obs_in_cat = self.store.list_observables(domain=domain, category=category)

                if fl and not self._category_matches(fl, category, obs_in_cat):
                    continue

                expand_cat = bool(fl)
                n_cat = n_dom.add(
                    f"📂 {category}",
                    data=_CategoryMeta(domain, category),
                    expand=expand_cat,
                )

                for obs in obs_in_cat:
                    if fl and fl not in obs.classification.name.lower():
                        # Still show if parent category matches the filter
                        if fl not in category.lower():
                            continue

                    fd_cnt = len(obs.field_definitions)
                    dc_cnt = len(obs.dataset_classifications)
                    badges = (
                        (f" [F:{fd_cnt}]" if fd_cnt else "")
                        + (f" [D:{dc_cnt}]" if dc_cnt else "")
                    )
                    n_cat.add_leaf(
                        f"🔵 {obs.classification.name}{badges}",
                        data=_ObservableMeta(obs),
                    )

    @staticmethod
    def _domain_matches(fl: str, domain: str, cats, obs_all) -> bool:
        if fl in domain.lower():
            return True
        if any(fl in c.classification.category.lower() for c in cats):
            return True
        if any(fl in o.classification.name.lower() for o in obs_all):
            return True
        return False

    @staticmethod
    def _category_matches(fl: str, category: str, obs_in_cat) -> bool:
        if fl in category.lower():
            return True
        if any(fl in o.classification.name.lower() for o in obs_in_cat):
            return True
        return False

    # ------------------------------------------------------------------
    # Tree selection → detail panel
    # ------------------------------------------------------------------

    @on(Tree.NodeSelected, "#ontology-tree")
    def handle_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if isinstance(data, _ObservableMeta):
            self._show_observable(data.node)
        elif isinstance(data, _DomainMeta):
            self._show_domain(data.domain)
        elif isinstance(data, _CategoryMeta):
            self._show_category(data.domain, data.category)

    def _show_observable(self, node: OntologyNode) -> None:
        nc = node.classification
        lines: list[str] = [
            f"[bold cyan]{nc.classification_str}[/bold cyan]",
            f"[dim]{node.id}[/dim]",
        ]
        if node.description:
            lines += ["", node.description]

        # Field definitions
        if node.field_definitions:
            lines.append("\n[bold green]Field Definitions[/bold green]")
            for fd in node.field_definitions:
                dt = f" [dim]({fd.data_type})[/dim]" if fd.data_type else ""
                lines.append(f"  • [bold]{fd.name}[/bold]{dt}")
                if fd.aliases:
                    lines.append(f"    [dim]aliases:[/dim] {', '.join(fd.aliases)}")
                if fd.description:
                    lines.append(f"    [dim]{fd.description}[/dim]")
        else:
            lines.append("\n[dim](no field definitions)[/dim]")

        # Dataset classifications
        if node.dataset_classifications:
            lines.append("\n[bold green]Dataset Classifications[/bold green]")
            for dc in node.dataset_classifications:
                lines.append(f"  • [bold]{dc.dataset_name}[/bold]")
                if dc.source:
                    lines.append(f"    [dim]source:[/dim]  {dc.source}")
                if dc.dataset_type:
                    lines.append(f"    [dim]type:[/dim]    {dc.dataset_type}")
                if dc.description:
                    lines.append(f"    [dim]{dc.description}[/dim]")
                if dc.field_names:
                    lines.append(f"    [dim]fields:[/dim]  {', '.join(dc.field_names)}")
                if dc.aliases:
                    lines.append(f"    [dim]aliases:[/dim] {', '.join(dc.aliases)}")
        else:
            lines.append("\n[dim](no dataset classifications)[/dim]")

        # Confirmed mappings
        mappings = self.store.get_confirmed_mappings(node_id=node.id)
        if mappings:
            lines.append(
                f"\n[bold green]Confirmed Mappings[/bold green] [dim]({len(mappings)} total)[/dim]"
            )
            for m in mappings[:20]:
                conf_str = f" [dim][{m.confidence:.2f}][/dim]" if m.confidence is not None else ""
                canon_str = (
                    f" → [yellow]{m.canonical_field}[/yellow]" if m.canonical_field else ""
                )
                src_color = "cyan" if m.source == "auto" else "magenta"
                lines.append(
                    f"  • [dim]{m.created_at[:10]}[/dim]"
                    f" [{src_color}]{m.source}[/{src_color}]{conf_str}"
                    f"  [green]{m.dataset_field}[/green]{canon_str}"
                    f"  [dim]({m.dataset_name})[/dim]"
                )
                if m.notes:
                    lines.append(f"    [dim]↳ {m.notes}[/dim]")
            if len(mappings) > 20:
                lines.append(f"  [dim]… and {len(mappings) - 20} more[/dim]")

        self.query_one("#detail-content", Static).update("\n".join(lines))

    def _show_domain(self, domain: str) -> None:
        cats = self.store.list_categories(domain=domain)
        obs = self.store.list_observables(domain=domain)
        n_cats = len(cats)
        n_obs = len(obs)
        self.query_one("#detail-content", Static).update(
            f"[bold cyan]Domain: {domain}[/bold cyan]\n\n"
            f"  {n_cats} {'category' if n_cats == 1 else 'categories'}\n"
            f"  {n_obs} {'observable' if n_obs == 1 else 'observables'}\n\n"
            f"[dim]Expand the tree node to browse categories.[/dim]"
        )

    def _show_category(self, domain: str, category: str) -> None:
        obs = self.store.list_observables(domain=domain, category=category)
        n_obs = len(obs)
        lines = [
            f"[bold cyan]Category: {category}[/bold cyan]",
            f"[dim]Domain: {domain}[/dim]",
            f"\n  {n_obs} {'observable' if n_obs == 1 else 'observables'}",
        ]
        if obs:
            lines.append("")
            for o in obs:
                fd = len(o.field_definitions)
                dc = len(o.dataset_classifications)
                lines.append(
                    f"  🔵 {o.classification.name}"
                    + (f"  [dim][F:{fd}][/dim]" if fd else "")
                    + (f"  [dim][D:{dc}][/dim]" if dc else "")
                )
        lines.append("\n[dim]Select an observable node for full details.[/dim]")
        self.query_one("#detail-content", Static).update("\n".join(lines))

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    @on(Input.Changed, "#filter-input")
    def handle_filter_changed(self, event: Input.Changed) -> None:
        self._rebuild_tree(event.value)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#search-btn")
    def handle_search(self, _) -> None:
        ds_name = self.query_one("#ds-name-input", Input).value.strip() or None
        raw = self.query_one("#fields-input", TextArea).text.strip()
        field_names = [f.strip() for f in raw.splitlines() if f.strip()] or None
        domain = self.query_one("#domain-filter", Input).value.strip() or None

        try:
            threshold = float(self.query_one("#threshold-input", Input).value)
        except ValueError:
            threshold = 0.55

        if not ds_name and not field_names:
            self.query_one("#results-content", Static).update(
                "[red]Provide a dataset name and/or at least one field name.[/red]"
            )
            return

        self.searcher.threshold = threshold
        self._last_results = self.searcher.search(
            dataset_name=ds_name,
            field_names=field_names,
            domain=domain,
            top_k=20,
        )
        self._render_results()
        has = bool(self._last_results)
        self.query_one("#confirm-top-btn", Button).disabled = not has
        self.query_one("#confirm-all-btn", Button).disabled = not has
        self.query_one("#confirm-status", Static).update("")

    def _render_results(self) -> None:
        results = self._last_results
        if not results:
            self.query_one("#results-content", Static).update(
                "[dim]No matches found. Try lowering the threshold.[/dim]"
            )
            return

        lines: list[str] = [f"[bold]{len(results)} match(es):[/bold]\n"]

        for i, r in enumerate(results, 1):
            filled = int(r.combined_score * 12)
            bar = f"[cyan]{'█' * filled}[/cyan][dim]{'░' * (12 - filled)}[/dim]"
            lines.append(f"[bold][{i}][/bold] [cyan]{r.classification}[/cyan]")
            lines.append(
                f"    {bar} {r.combined_score:.2f}"
                f"  [dim]dataset={r.dataset_score:.2f}  fields={r.field_score:.2f}[/dim]"
            )
            if r.matched_dataset_names:
                lines.append(
                    f"    [dim]matched dataset:[/dim] {', '.join(r.matched_dataset_names)}"
                )
            for fm in r.field_matches:
                icon = "[green]✓[/green]" if fm.score >= 0.8 else "[yellow]~[/yellow]"
                lines.append(
                    f"    {icon} [green]{fm.query_field!r}[/green]"
                    f" → [yellow]{fm.matched_field!r}[/yellow]"
                    f" [dim]({fm.score:.2f})[/dim]"
                )
            if r.unmatched_fields:
                lines.append(
                    f"    [red]✗ unmatched:[/red] {', '.join(r.unmatched_fields)}"
                )
            lines.append("")

        self.query_one("#results-content", Static).update("\n".join(lines))

    # ------------------------------------------------------------------
    # Confirm
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#confirm-top-btn")
    def handle_confirm_top(self, _) -> None:
        self._confirm(self._last_results[:1])

    @on(Button.Pressed, "#confirm-all-btn")
    def handle_confirm_all(self, _) -> None:
        self._confirm(self._last_results)

    def _confirm(self, results: list) -> None:
        if not results:
            return
        ds_name = self.query_one("#ds-name-input", Input).value.strip()
        if not ds_name:
            self.query_one("#confirm-status", Static).update(
                "[yellow]⚠ Enter a dataset name before confirming.[/yellow]"
            )
            return

        total = 0
        for result in results:
            confirmed = self.searcher.confirm_result(
                result, dataset_name=ds_name, source="auto"
            )
            total += len(confirmed)

        self.query_one("#confirm-status", Static).update(
            f"[green]✓ Saved {total} mapping(s) for {len(results)} observable(s).[/green]"
        )
        # Refresh tree badges to reflect new D/F counts
        self._rebuild_tree(self.query_one("#filter-input", Input).value)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_refresh(self) -> None:
        self._rebuild_tree(self.query_one("#filter-input", Input).value)
        self.notify("Tree refreshed.")

    def action_focus_filter(self) -> None:
        self.query_one("#filter-input", Input).focus()

    def on_unmount(self) -> None:
        self.store.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--db",
    default="ontology.db",
    show_default=True,
    envvar="ONTOLOGY_DB",
    help="Path to the SQLite database file.",
)
def main(db: str) -> None:
    """Launch the interactive TUI for the ontological classification system."""
    OntologyBrowser(db_path=db).run()


if __name__ == "__main__":
    main()
