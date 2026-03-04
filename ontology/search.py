"""
Dataset search and matching engine.

Given an incoming dataset (name and/or column names), this module scores every
observable in the ontology and returns ranked candidates.

Matching strategy
-----------------
1. **Token normalisation** – underscore, hyphen, and dot separators are replaced
   with spaces before comparison so that ``total_population``,
   ``total-population``, and ``total population`` all score identically.

2. **Sequence similarity** – Python's ``difflib.SequenceMatcher`` provides a
   character-level similarity ratio (0 – 1).

3. **Token overlap** – Jaccard similarity on word tokens catches partial matches
   that sequence similarity misses (e.g. ``pop_estimate`` vs ``population_estimate``).

4. **Combined field score** – for each incoming field name the best match across
   *all* known field names of an observable is found.  The final field score
   weights both match quality and coverage (how many incoming fields were matched).

5. **Combined dataset score** – the incoming dataset name is matched against the
   known dataset names and aliases attached to the observable.

6. **Overall score** – weighted combination of dataset score (40 %) and field
   score (60 %) when both are supplied; otherwise whichever is present.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Optional

from .models import OntologyNode
from .store import OntologyStore


# ---------------------------------------------------------------------------
# String similarity helpers
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    """Lower-case and replace common separators with spaces."""
    return s.lower().replace("_", " ").replace("-", " ").replace(".", " ").strip()


def _token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on whitespace-split tokens."""
    ta = set(_normalise(a).split())
    tb = set(_normalise(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _seq_ratio(a: str, b: str) -> float:
    na, nb = _normalise(a), _normalise(b)
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def similarity(a: str, b: str) -> float:
    """
    Combined similarity score between two strings (0 – 1).

    Takes the maximum of sequence ratio and token Jaccard so that both
    character-level and word-level similarity are captured.
    """
    return max(_seq_ratio(a, b), _token_jaccard(a, b))


def best_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.55,
) -> Optional[tuple[str, float]]:
    """
    Find the best-matching candidate for *query*.

    Returns ``(best_candidate, score)`` or ``None`` if no candidate exceeds
    *threshold*.
    """
    best: Optional[str] = None
    best_score = threshold
    for candidate in candidates:
        score = similarity(query, candidate)
        if score > best_score:
            best_score = score
            best = candidate
    return (best, best_score) if best is not None else None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FieldMatchDetail:
    """A single field-name match."""
    query_field: str
    matched_field: str
    score: float


@dataclass
class DatasetSearchResult:
    """
    Scoring result for one observable against an incoming dataset.

    Attributes
    ----------
    observable:
        The matched OntologyNode.
    dataset_score:
        Confidence that the incoming dataset name maps to this observable (0–1).
    field_score:
        Confidence based on field-name overlap (0–1).
    combined_score:
        Weighted combination used for ranking (0–1).
    matched_dataset_names:
        Which known dataset names / aliases were matched.
    field_matches:
        Per-field match details for matched fields.
    unmatched_fields:
        Incoming fields with no match above threshold.
    suggested_mapping:
        Dict mapping each incoming field name to the canonical field definition
        name it best matched (``None`` for unmatched fields).
    """
    observable: OntologyNode
    dataset_score: float
    field_score: float
    combined_score: float
    matched_dataset_names: list[str] = field(default_factory=list)
    field_matches: list[FieldMatchDetail] = field(default_factory=list)
    unmatched_fields: list[str] = field(default_factory=list)
    suggested_mapping: dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def classification(self) -> str:
        return self.observable.classification.classification_str

    def summary(self) -> str:
        lines = [
            f"  {self.classification}",
            f"  score={self.combined_score:.2f}"
            f"  (dataset={self.dataset_score:.2f}, fields={self.field_score:.2f})",
        ]
        if self.matched_dataset_names:
            lines.append(f"  matched datasets: {', '.join(self.matched_dataset_names)}")
        if self.field_matches:
            for fm in self.field_matches:
                lines.append(
                    f"    {fm.query_field!r} -> {fm.matched_field!r} ({fm.score:.2f})"
                )
        if self.unmatched_fields:
            lines.append(f"  unmatched fields: {', '.join(self.unmatched_fields)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class OntologySearcher:
    """
    High-level search interface over an OntologyStore.

    Parameters
    ----------
    store:
        The backing store to search.
    threshold:
        Minimum similarity score (0–1) for a match to be considered.
        Lower values return more (possibly noisy) candidates.
    dataset_weight:
        Weight given to the dataset-name score in the combined score.
        Field-name score weight = 1 - dataset_weight.
    """

    def __init__(
        self,
        store: OntologyStore,
        threshold: float = 0.55,
        dataset_weight: float = 0.40,
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.dataset_weight = dataset_weight
        self._field_weight = 1.0 - dataset_weight

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def search(
        self,
        dataset_name: Optional[str] = None,
        field_names: Optional[list[str]] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 10,
    ) -> list[DatasetSearchResult]:
        """
        Search observables for the best matches to an incoming dataset.

        At least one of *dataset_name* or *field_names* must be provided.

        Parameters
        ----------
        dataset_name:
            Name (or any alias) of the incoming dataset.
        field_names:
            Column/field names present in the incoming dataset.
        domain:
            Restrict search to this domain.
        category:
            Restrict search to this category.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List of DatasetSearchResult sorted by combined_score descending.
        """
        if not dataset_name and not field_names:
            raise ValueError("Provide at least dataset_name or field_names.")

        observables = self.store.list_observables(domain=domain, category=category)
        results: list[DatasetSearchResult] = []

        for obs in observables:
            result = self._score(obs, dataset_name, field_names or [])
            if result.combined_score > 0:
                results.append(result)

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    def suggest_field_mappings(
        self,
        field_names: list[str],
        domain: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> dict[str, list[DatasetSearchResult]]:
        """
        For each incoming field name, independently return the best-matching
        observables.

        Useful when you want per-column classification suggestions rather than
        a single dataset-level result.

        Returns
        -------
        Dict mapping each field name to a ranked list of DatasetSearchResult.
        """
        observables = self.store.list_observables(domain=domain, category=category)
        suggestions: dict[str, list[DatasetSearchResult]] = {}

        for qf in field_names:
            field_results: list[DatasetSearchResult] = []
            for obs in observables:
                known = obs.all_field_names()
                if not known:
                    continue
                match = best_match(qf, known, self.threshold)
                if match:
                    matched_name, score = match
                    # Find the canonical name from field_definitions
                    canonical = self._resolve_canonical(obs, matched_name)
                    result = DatasetSearchResult(
                        observable=obs,
                        dataset_score=0.0,
                        field_score=score,
                        combined_score=score,
                        field_matches=[FieldMatchDetail(qf, matched_name, score)],
                        suggested_mapping={qf: canonical},
                    )
                    field_results.append(result)

            field_results.sort(key=lambda r: r.combined_score, reverse=True)
            suggestions[qf] = field_results[:top_k]

        return suggestions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score(
        self,
        obs: OntologyNode,
        dataset_name: Optional[str],
        field_names: list[str],
    ) -> DatasetSearchResult:
        # --- Dataset name score ---
        dataset_score = 0.0
        matched_dataset_names: list[str] = []

        if dataset_name:
            known_names: list[str] = []
            for dc in obs.dataset_classifications:
                known_names.extend(dc.all_dataset_names())
            if known_names:
                match = best_match(dataset_name, known_names, self.threshold)
                if match:
                    matched_dataset_names.append(match[0])
                    dataset_score = match[1]

        # --- Field name score ---
        field_score = 0.0
        field_matches: list[FieldMatchDetail] = []
        unmatched_fields: list[str] = []
        suggested_mapping: dict[str, Optional[str]] = {}

        if field_names:
            known_fields = obs.all_field_names()
            if known_fields:
                total_score = 0.0
                for qf in field_names:
                    match = best_match(qf, known_fields, self.threshold)
                    if match:
                        matched_name, score = match
                        canonical = self._resolve_canonical(obs, matched_name)
                        field_matches.append(FieldMatchDetail(qf, matched_name, score))
                        suggested_mapping[qf] = canonical
                        total_score += score
                    else:
                        unmatched_fields.append(qf)
                        suggested_mapping[qf] = None

                if field_matches:
                    avg_score = total_score / len(field_names)
                    coverage = len(field_matches) / len(field_names)
                    # Penalise low coverage: good avg score with few matches
                    # should score lower than good avg score with many matches.
                    field_score = avg_score * (0.5 + 0.5 * coverage)
            else:
                unmatched_fields = list(field_names)
                suggested_mapping = {qf: None for qf in field_names}

        # --- Combined score ---
        has_ds = dataset_name is not None
        has_fs = bool(field_names)

        if has_ds and has_fs:
            combined_score = (
                self.dataset_weight * dataset_score
                + self._field_weight * field_score
            )
        elif has_ds:
            combined_score = dataset_score
        else:
            combined_score = field_score

        return DatasetSearchResult(
            observable=obs,
            dataset_score=dataset_score,
            field_score=field_score,
            combined_score=combined_score,
            matched_dataset_names=matched_dataset_names,
            field_matches=field_matches,
            unmatched_fields=unmatched_fields,
            suggested_mapping=suggested_mapping,
        )

    @staticmethod
    def _resolve_canonical(obs: OntologyNode, matched_name: str) -> Optional[str]:
        """
        Given a matched field name (could be an alias or dataset-specific name),
        return the canonical FieldDefinition name if one exists.
        """
        for fd in obs.field_definitions:
            if matched_name in fd.all_names():
                return fd.name
        # Matched a dataset-specific field name with no canonical definition
        return matched_name
