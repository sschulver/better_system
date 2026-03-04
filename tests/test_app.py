"""
Smoke tests for the TUI app.

These tests verify that the app composes correctly, the tree builds from live
store data, the filter narrows results, and the search + confirm flow works
end-to-end — all without launching an interactive terminal session.
"""
import pytest

from app import OntologyBrowser, _DomainMeta, _CategoryMeta, _ObservableMeta
from ontology import OntologyStore, OntologyNode
from ontology.models import (
    NodeClassification,
    FieldDefinition,
    DatasetClassification,
)
from textual.widgets import Tree, Input, Static


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_store(store: OntologyStore) -> None:
    store.add_node(OntologyNode(
        classification=NodeClassification.parse("concept:demography:*:domain:demography"),
        description="Demographic domain",
    ))
    store.add_node(OntologyNode(
        classification=NodeClassification.parse(
            "concept:demography:population-distribution:category:population-distribution"
        ),
    ))
    store.add_node(OntologyNode(
        classification=NodeClassification.parse(
            "observable:demography:population-distribution"
            ":spatial-aggregate:population-spatial-aggregate"
        ),
        description="Spatially aggregated population",
        field_definitions=[
            FieldDefinition(
                name="total_population",
                aliases=["pop_total", "population"],
                data_type="integer",
            ),
            FieldDefinition(
                name="population_density",
                aliases=["pop_density", "density"],
                data_type="float",
            ),
        ],
        dataset_classifications=[
            DatasetClassification(
                dataset_name="American Community Survey",
                source="US Census Bureau",
                dataset_type="survey",
                field_names=["B01003_001E", "total_pop"],
                aliases=["ACS", "ACS5"],
            ),
        ],
    ))
    store.add_node(OntologyNode(
        classification=NodeClassification.parse("concept:economics:*:domain:economics"),
    ))
    store.add_node(OntologyNode(
        classification=NodeClassification.parse("concept:economics:income:category:income"),
    ))
    store.add_node(OntologyNode(
        classification=NodeClassification.parse(
            "observable:economics:income:aggregate:median-household-income"
        ),
        field_definitions=[
            FieldDefinition(name="median_income", aliases=["med_income"]),
        ],
    ))


@pytest.fixture
def seeded_app():
    app = OntologyBrowser(db_path=":memory:")
    _seed_store(app.store)
    yield app
    app.store.close()


# ---------------------------------------------------------------------------
# Instantiation (no async needed)
# ---------------------------------------------------------------------------

class TestAppInstantiation:
    def test_app_creates_without_error(self):
        app = OntologyBrowser(db_path=":memory:")
        assert app is not None
        app.store.close()

    def test_app_store_is_accessible(self, seeded_app):
        assert len(seeded_app.store.list_all()) == 6


# ---------------------------------------------------------------------------
# Tree-building static helpers (no async needed)
# ---------------------------------------------------------------------------

class TestTreeBuilding:
    def test_domain_matches_by_name(self):
        assert OntologyBrowser._domain_matches("demography", "demography", [], [])

    def test_domain_matches_by_category(self):
        cat = OntologyNode(
            classification=NodeClassification.parse(
                "concept:demography:population-distribution:category:population-distribution"
            )
        )
        assert OntologyBrowser._domain_matches("population", "demography", [cat], [])

    def test_domain_matches_by_observable(self):
        obs = OntologyNode(
            classification=NodeClassification.parse(
                "observable:demography:population-distribution"
                ":spatial-aggregate:population-spatial-aggregate"
            )
        )
        assert OntologyBrowser._domain_matches("spatial", "demography", [], [obs])

    def test_domain_no_match(self):
        assert not OntologyBrowser._domain_matches("xyz", "demography", [], [])

    def test_category_matches_by_name(self):
        assert OntologyBrowser._category_matches("income", "income", [])

    def test_category_matches_by_observable(self):
        obs = OntologyNode(
            classification=NodeClassification.parse(
                "observable:economics:income:aggregate:median-household-income"
            )
        )
        assert OntologyBrowser._category_matches("median", "income", [obs])

    def test_category_no_match(self):
        assert not OntologyBrowser._category_matches("xyz", "income", [])


# ---------------------------------------------------------------------------
# TUI integration — async pilot tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_app_mounts_without_error(seeded_app):
    async with seeded_app.run_test(headless=True) as pilot:
        assert seeded_app.query_one("#ontology-tree") is not None


@pytest.mark.asyncio
async def test_tree_has_domain_nodes(seeded_app):
    async with seeded_app.run_test(headless=True) as pilot:
        tree = seeded_app.query_one("#ontology-tree", Tree)
        domain_nodes = list(tree.root.children)
        assert len(domain_nodes) == 2


@pytest.mark.asyncio
async def test_tree_domain_node_has_meta(seeded_app):
    async with seeded_app.run_test(headless=True) as pilot:
        tree = seeded_app.query_one("#ontology-tree", Tree)
        first_dom = list(tree.root.children)[0]
        assert isinstance(first_dom.data, _DomainMeta)


@pytest.mark.asyncio
async def test_filter_narrows_tree(seeded_app):
    """Setting the filter input value triggers _rebuild_tree via Input.Changed."""
    async with seeded_app.run_test(headless=True) as pilot:
        filter_input = seeded_app.query_one("#filter-input", Input)
        # Setting .value fires Input.Changed which calls _rebuild_tree
        filter_input.value = "demography"
        await pilot.pause()
        tree = seeded_app.query_one("#ontology-tree", Tree)
        domain_nodes = list(tree.root.children)
        assert len(domain_nodes) == 1
        assert domain_nodes[0].data.domain == "demography"


@pytest.mark.asyncio
async def test_filter_clear_restores_all(seeded_app):
    async with seeded_app.run_test(headless=True) as pilot:
        filter_input = seeded_app.query_one("#filter-input", Input)
        filter_input.value = "demography"
        await pilot.pause()
        filter_input.value = ""
        await pilot.pause()
        tree = seeded_app.query_one("#ontology-tree", Tree)
        assert len(list(tree.root.children)) == 2


@pytest.mark.asyncio
async def test_filter_observable_name(seeded_app):
    async with seeded_app.run_test(headless=True) as pilot:
        filter_input = seeded_app.query_one("#filter-input", Input)
        filter_input.value = "median"
        await pilot.pause()
        tree = seeded_app.query_one("#ontology-tree", Tree)
        # Only the economics domain should remain
        domain_nodes = list(tree.root.children)
        assert len(domain_nodes) == 1
        assert domain_nodes[0].data.domain == "economics"


@pytest.mark.asyncio
async def test_rebuild_tree_adds_new_domain(seeded_app):
    """Calling action_refresh picks up nodes added after initial mount."""
    async with seeded_app.run_test(headless=True) as pilot:
        seeded_app.store.add_node(OntologyNode(
            classification=NodeClassification.parse("concept:health:*:domain:health"),
        ))
        seeded_app.action_refresh()
        await pilot.pause()
        tree = seeded_app.query_one("#ontology-tree", Tree)
        assert len(list(tree.root.children)) == 3


@pytest.mark.asyncio
async def test_confirm_requires_dataset_name(seeded_app):
    """_confirm() with empty dataset name posts a warning, not a crash."""
    async with seeded_app.run_test(headless=True) as pilot:
        results = seeded_app.searcher.search(field_names=["total_population"])
        seeded_app._last_results = results
        # Leave ds-name-input empty
        seeded_app._confirm(results)
        await pilot.pause()
        status = seeded_app.query_one("#confirm-status", Static)
        text = status.content
        assert "dataset name" in text.lower() or "⚠" in text


@pytest.mark.asyncio
async def test_confirm_top_saves_mapping(seeded_app):
    """_confirm() with a dataset name persists ConfirmedMapping rows."""
    async with seeded_app.run_test(headless=True) as pilot:
        results = seeded_app.searcher.search(field_names=["total_population"])
        seeded_app._last_results = results
        seeded_app.query_one("#ds-name-input", Input).value = "Test Dataset 2024"
        await pilot.pause()
        seeded_app._confirm(results[:1])
        await pilot.pause()
        node_id = results[0].observable.id
        mappings = seeded_app.store.get_confirmed_mappings(node_id=node_id)
        assert len(mappings) >= 1
        assert mappings[0].dataset_name == "Test Dataset 2024"


@pytest.mark.asyncio
async def test_confirm_all_saves_multiple(seeded_app):
    """_confirm() over all results saves mappings for each observable."""
    async with seeded_app.run_test(headless=True) as pilot:
        results = seeded_app.searcher.search(field_names=["total_population", "density"])
        seeded_app._last_results = results
        seeded_app.query_one("#ds-name-input", Input).value = "Multi DS"
        await pilot.pause()
        seeded_app._confirm(results)
        await pilot.pause()
        all_mappings = seeded_app.store.get_confirmed_mappings(dataset_name="Multi DS")
        assert len(all_mappings) >= 1


@pytest.mark.asyncio
async def test_observable_detail_rendered(seeded_app):
    """Selecting an observable node populates the detail panel."""
    async with seeded_app.run_test(headless=True) as pilot:
        tree = seeded_app.query_one("#ontology-tree", Tree)
        # Expand domain → category to reach the observable leaf
        dom_node = list(tree.root.children)[0]        # demography (sorted first)
        dom_node.expand()
        await pilot.pause()
        cat_node = list(dom_node.children)[0]
        cat_node.expand()
        await pilot.pause()
        obs_node = list(cat_node.children)[0]
        # Simulate selection
        seeded_app._show_observable(obs_node.data.node)
        await pilot.pause()
        detail = seeded_app.query_one("#detail-content", Static)
        text = detail.content
        assert "population-spatial-aggregate" in text
        assert "total_population" in text


@pytest.mark.asyncio
async def test_domain_detail_rendered(seeded_app):
    async with seeded_app.run_test(headless=True) as pilot:
        seeded_app._show_domain("demography")
        await pilot.pause()
        detail = seeded_app.query_one("#detail-content", Static)
        text = detail.content
        assert "demography" in text.lower()


@pytest.mark.asyncio
async def test_category_detail_rendered(seeded_app):
    async with seeded_app.run_test(headless=True) as pilot:
        seeded_app._show_category("demography", "population-distribution")
        await pilot.pause()
        detail = seeded_app.query_one("#detail-content", Static)
        text = detail.content
        assert "population-distribution" in text
