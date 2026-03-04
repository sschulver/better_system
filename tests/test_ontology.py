"""
Tests for the ontological classification system.
"""
import pytest

from ontology.models import (
    DatasetClassification,
    FieldDefinition,
    NodeClassification,
    OntologyNode,
)
from ontology.store import OntologyStore
from ontology.search import OntologySearcher, similarity, best_match


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    s = OntologyStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store pre-loaded with demography domain, category, and observable."""
    store.add_node(_domain_node())
    store.add_node(_category_node())
    store.add_node(_observable_node())
    return store


def _domain_node() -> OntologyNode:
    return OntologyNode(
        classification=NodeClassification.parse(
            "concept:demography:*:domain:demography"
        ),
        description="Demographic domain covering population and related concepts",
    )


def _category_node() -> OntologyNode:
    return OntologyNode(
        classification=NodeClassification.parse(
            "concept:demography:population-distribution:category:population-distribution"
        ),
        description="Population distribution and spatial patterns",
    )


def _observable_node() -> OntologyNode:
    return OntologyNode(
        classification=NodeClassification.parse(
            "observable:demography:population-distribution"
            ":spatial-aggregate:population-spatial-aggregate"
        ),
        description="Spatially aggregated population counts",
        field_definitions=[
            FieldDefinition(
                name="total_population",
                aliases=["pop_total", "population", "total_pop", "pop"],
                data_type="integer",
                description="Total population count",
            ),
            FieldDefinition(
                name="population_density",
                aliases=["pop_density", "density"],
                data_type="float",
                description="Population per unit area",
            ),
        ],
        dataset_classifications=[
            DatasetClassification(
                dataset_name="US Census Bureau Decennial Census",
                source="US Census Bureau",
                dataset_type="census",
                field_names=["P001001", "total_population", "POPESTIMATE"],
                aliases=["decennial_census", "census_2020", "census_2010"],
            ),
            DatasetClassification(
                dataset_name="American Community Survey",
                source="US Census Bureau",
                dataset_type="survey",
                field_names=["B01003_001E", "total_pop", "population_estimate"],
                aliases=["ACS", "ACS5", "acs_5yr"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# NodeClassification
# ---------------------------------------------------------------------------

class TestNodeClassification:
    def test_parse_domain(self):
        nc = NodeClassification.parse("concept:demography:*:domain:demography")
        assert nc.object_class == "concept"
        assert nc.domain == "demography"
        assert nc.category == "*"
        assert nc.type == "domain"
        assert nc.name == "demography"

    def test_parse_category(self):
        nc = NodeClassification.parse(
            "concept:demography:population-distribution:category:population-distribution"
        )
        assert nc.object_class == "concept"
        assert nc.type == "category"
        assert nc.category == "population-distribution"

    def test_parse_observable(self):
        nc = NodeClassification.parse(
            "observable:demography:population-distribution"
            ":spatial-aggregate:population-spatial-aggregate"
        )
        assert nc.object_class == "observable"
        assert nc.type == "spatial-aggregate"

    def test_classification_str_roundtrip(self):
        s = "concept:demography:*:domain:demography"
        assert NodeClassification.parse(s).classification_str == s

    def test_str(self):
        nc = NodeClassification.parse("concept:demography:*:domain:demography")
        assert str(nc) == "concept:demography:*:domain:demography"

    def test_invalid_classification_raises(self):
        with pytest.raises(ValueError, match="Invalid classification"):
            NodeClassification.parse("concept:demography")

    def test_invalid_too_many_parts(self):
        # Extra colons in name are captured by split(maxsplit=4), so 5 parts
        # with a colon in name should still parse correctly.
        nc = NodeClassification.parse("concept:demography:*:domain:demo:extra")
        assert nc.name == "demo:extra"


# ---------------------------------------------------------------------------
# OntologyNode helpers
# ---------------------------------------------------------------------------

class TestOntologyNode:
    def test_is_domain(self):
        assert _domain_node().is_domain
        assert not _domain_node().is_observable

    def test_is_category(self):
        assert _category_node().is_category

    def test_is_observable(self):
        obs = _observable_node()
        assert obs.is_observable
        assert not obs.is_domain
        assert not obs.is_category

    def test_all_field_names_includes_aliases_and_dataset_fields(self):
        obs = _observable_node()
        names = obs.all_field_names()
        # canonical
        assert "total_population" in names
        # alias
        assert "pop_total" in names
        # dataset-specific
        assert "B01003_001E" in names
        assert "P001001" in names


# ---------------------------------------------------------------------------
# OntologyStore
# ---------------------------------------------------------------------------

class TestOntologyStore:
    def test_add_and_get_by_id(self, store):
        node = _domain_node()
        store.add_node(node)
        retrieved = store.get_node(node.id)
        assert retrieved is not None
        assert retrieved.id == node.id
        assert retrieved.classification.domain == "demography"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_node("nonexistent-id") is None

    def test_get_by_classification(self, store):
        node = _domain_node()
        store.add_node(node)
        retrieved = store.get_node_by_classification(
            "concept:demography:*:domain:demography"
        )
        assert retrieved is not None
        assert retrieved.id == node.id

    def test_get_by_classification_not_found(self, store):
        assert (
            store.get_node_by_classification("concept:missing:*:domain:missing") is None
        )

    def test_add_node_preserves_field_definitions(self, store):
        obs = _observable_node()
        store.add_node(obs)
        retrieved = store.get_node(obs.id)
        assert len(retrieved.field_definitions) == 2
        names = {fd.name for fd in retrieved.field_definitions}
        assert "total_population" in names
        assert "population_density" in names

    def test_add_node_preserves_aliases(self, store):
        obs = _observable_node()
        store.add_node(obs)
        retrieved = store.get_node(obs.id)
        pop_fd = next(fd for fd in retrieved.field_definitions if fd.name == "total_population")
        assert "pop_total" in pop_fd.aliases
        assert "total_pop" in pop_fd.aliases

    def test_add_node_preserves_dataset_classifications(self, store):
        obs = _observable_node()
        store.add_node(obs)
        retrieved = store.get_node(obs.id)
        assert len(retrieved.dataset_classifications) == 2
        ds_names = {dc.dataset_name for dc in retrieved.dataset_classifications}
        assert "American Community Survey" in ds_names
        assert "US Census Bureau Decennial Census" in ds_names

    def test_add_node_preserves_dataset_field_names(self, store):
        obs = _observable_node()
        store.add_node(obs)
        retrieved = store.get_node(obs.id)
        acs = next(
            dc for dc in retrieved.dataset_classifications
            if dc.dataset_name == "American Community Survey"
        )
        assert "B01003_001E" in acs.field_names
        assert "total_pop" in acs.field_names

    def test_add_node_preserves_dataset_aliases(self, store):
        obs = _observable_node()
        store.add_node(obs)
        retrieved = store.get_node(obs.id)
        acs = next(
            dc for dc in retrieved.dataset_classifications
            if dc.dataset_name == "American Community Survey"
        )
        assert "ACS" in acs.aliases
        assert "ACS5" in acs.aliases

    def test_replace_node_clears_stale_children(self, store):
        obs = _observable_node()
        store.add_node(obs)

        # Replace with a version that has no field definitions
        obs_v2 = OntologyNode(
            id=obs.id,
            classification=obs.classification,
            description="updated",
            field_definitions=[],
            dataset_classifications=[],
        )
        store.add_node(obs_v2)
        retrieved = store.get_node(obs.id)
        assert len(retrieved.field_definitions) == 0
        assert len(retrieved.dataset_classifications) == 0

    def test_update_description(self, store):
        node = _domain_node()
        store.add_node(node)
        store.update_description(node.id, "New description")
        retrieved = store.get_node(node.id)
        assert retrieved.description == "New description"

    def test_delete_node(self, store):
        node = _domain_node()
        store.add_node(node)
        assert store.delete_node(node.id)
        assert store.get_node(node.id) is None

    def test_delete_nonexistent_returns_false(self, store):
        assert not store.delete_node("nonexistent")

    def test_add_field_definition(self, store):
        obs = _observable_node()
        store.add_node(obs)
        new_fd = FieldDefinition(name="median_age", aliases=["age_median"], data_type="float")
        store.add_field_definition(obs.id, new_fd)
        retrieved = store.get_node(obs.id)
        names = {fd.name for fd in retrieved.field_definitions}
        assert "median_age" in names

    def test_add_dataset_classification(self, store):
        obs = _observable_node()
        store.add_node(obs)
        new_dc = DatasetClassification(
            dataset_name="WorldPop",
            source="WorldPop Project",
            field_names=["population", "pop_count"],
        )
        store.add_dataset_classification(obs.id, new_dc)
        retrieved = store.get_node(obs.id)
        ds_names = {dc.dataset_name for dc in retrieved.dataset_classifications}
        assert "WorldPop" in ds_names

    def test_list_domains(self, populated_store):
        domains = populated_store.list_domains()
        assert len(domains) == 1
        assert domains[0].classification.domain == "demography"

    def test_list_categories(self, populated_store):
        cats = populated_store.list_categories()
        assert len(cats) == 1
        assert cats[0].classification.type == "category"

    def test_list_categories_by_domain(self, populated_store):
        cats = populated_store.list_categories(domain="demography")
        assert len(cats) == 1

    def test_list_categories_wrong_domain(self, populated_store):
        cats = populated_store.list_categories(domain="economics")
        assert len(cats) == 0

    def test_list_observables(self, populated_store):
        obs = populated_store.list_observables()
        assert len(obs) == 1
        assert obs[0].is_observable

    def test_list_observables_by_domain(self, populated_store):
        obs = populated_store.list_observables(domain="demography")
        assert len(obs) == 1

    def test_list_observables_by_category(self, populated_store):
        obs = populated_store.list_observables(category="population-distribution")
        assert len(obs) == 1

    def test_list_observables_no_match(self, populated_store):
        obs = populated_store.list_observables(domain="economics")
        assert len(obs) == 0

    def test_search_nodes_by_name(self, populated_store):
        nodes = populated_store.search_nodes("demography")
        assert len(nodes) >= 1

    def test_list_all(self, populated_store):
        all_nodes = populated_store.list_all()
        assert len(all_nodes) == 3

    def test_context_manager(self):
        with OntologyStore(":memory:") as s:
            s.add_node(_domain_node())
            assert len(s.list_domains()) == 1
        # Connection closed after exit; further use would raise


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("total_population", "total_population") == 1.0

    def test_separator_normalisation(self):
        # underscore vs space vs hyphen should all score highly
        assert similarity("total_population", "total population") >= 0.9
        assert similarity("total-population", "total population") >= 0.9

    def test_case_insensitive(self):
        assert similarity("Population", "population") == 1.0

    def test_partial_match(self):
        score = similarity("pop_total", "total_population")
        assert 0.3 <= score <= 1.0

    def test_unrelated_strings(self):
        assert similarity("population", "revenue") < 0.5

    def test_best_match_found(self):
        result = best_match("total_pop", ["population", "total_population", "density"])
        assert result is not None
        matched, score = result
        assert matched == "total_population"
        assert score >= 0.6

    def test_best_match_none_below_threshold(self):
        result = best_match("revenue", ["population", "density"], threshold=0.8)
        assert result is None

    def test_best_match_alias(self):
        result = best_match("ACS5", ["American Community Survey", "ACS5", "decennial"])
        assert result is not None
        assert result[0] == "ACS5"
        assert result[1] == 1.0


# ---------------------------------------------------------------------------
# OntologySearcher
# ---------------------------------------------------------------------------

class TestOntologySearcher:
    @pytest.fixture
    def searcher(self, populated_store):
        return OntologySearcher(populated_store, threshold=0.5)

    def test_search_by_field_names_exact(self, searcher):
        results = searcher.search(field_names=["total_population", "population_density"])
        assert len(results) > 0
        assert results[0].field_score >= 0.7

    def test_search_by_field_alias(self, searcher):
        results = searcher.search(field_names=["pop_total"])
        assert len(results) > 0
        assert results[0].field_score > 0

    def test_search_by_dataset_field_name(self, searcher):
        # B01003_001E is a dataset-specific field name
        results = searcher.search(field_names=["B01003_001E"])
        assert len(results) > 0

    def test_search_by_dataset_name_exact(self, searcher):
        results = searcher.search(dataset_name="American Community Survey")
        assert len(results) > 0
        assert results[0].dataset_score >= 0.9

    def test_search_by_dataset_alias(self, searcher):
        results = searcher.search(dataset_name="ACS5")
        assert len(results) > 0
        assert results[0].dataset_score == 1.0

    def test_search_by_dataset_fuzzy(self, searcher):
        results = searcher.search(dataset_name="American Community Surveys")
        assert len(results) > 0

    def test_combined_search(self, searcher):
        results = searcher.search(
            dataset_name="ACS",
            field_names=["total_pop", "pop_density"],
        )
        assert len(results) > 0
        r = results[0]
        assert r.dataset_score > 0
        assert r.field_score > 0
        assert r.combined_score > 0

    def test_suggested_mapping_canonical(self, searcher):
        results = searcher.search(field_names=["pop_total"])
        assert len(results) > 0
        r = results[0]
        # pop_total is an alias for total_population
        assert r.suggested_mapping.get("pop_total") == "total_population"

    def test_unmatched_fields_tracked(self, searcher):
        results = searcher.search(field_names=["total_population", "completely_unrelated_xyz"])
        assert len(results) > 0
        r = results[0]
        assert "completely_unrelated_xyz" in r.unmatched_fields

    def test_no_results_below_threshold(self, populated_store):
        searcher = OntologySearcher(populated_store, threshold=0.99)
        results = searcher.search(field_names=["something_completely_different"])
        assert len(results) == 0

    def test_search_requires_at_least_one_input(self, searcher):
        with pytest.raises(ValueError):
            searcher.search()

    def test_suggest_field_mappings(self, searcher):
        suggestions = searcher.suggest_field_mappings(["pop_total", "density"])
        assert "pop_total" in suggestions
        assert "density" in suggestions
        assert len(suggestions["pop_total"]) > 0
        assert len(suggestions["density"]) > 0

    def test_suggest_field_mappings_no_match(self, searcher):
        suggestions = searcher.suggest_field_mappings(["xyz_completely_unknown_field"])
        assert "xyz_completely_unknown_field" in suggestions
        assert len(suggestions["xyz_completely_unknown_field"]) == 0

    def test_domain_filter(self, searcher):
        results = searcher.search(field_names=["total_population"], domain="demography")
        assert len(results) > 0

    def test_domain_filter_no_match(self, searcher):
        results = searcher.search(field_names=["total_population"], domain="economics")
        assert len(results) == 0

    def test_top_k(self, searcher):
        # With only one observable we can't test limit, but ensure it doesn't error
        results = searcher.search(field_names=["total_population"], top_k=1)
        assert len(results) <= 1

    def test_matched_dataset_names_populated(self, searcher):
        results = searcher.search(dataset_name="American Community Survey")
        assert len(results) > 0
        assert len(results[0].matched_dataset_names) > 0

    def test_result_summary_str(self, searcher):
        results = searcher.search(field_names=["total_population"])
        assert len(results) > 0
        summary = results[0].summary()
        assert "observable:demography" in summary
        assert "score=" in summary
