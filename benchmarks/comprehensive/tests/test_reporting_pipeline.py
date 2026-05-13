"""Tests for the reporting pipeline.

Covers result-store loading/normalization, typed missing reasons,
report-model rendering (Markdown/HTML/JSON), lint checks, golden
snapshot tests, section numbering stability, and wide-table routing.
"""
from __future__ import annotations
import json, textwrap, tempfile
from pathlib import Path
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FIXTURES = Path(__file__).parent / "fixtures"

def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from benchmarks.comprehensive.reporting.result_store import (
    ResultStore, ResultRow, SourceRef, SourceKind,
    MissingReason, ScenarioMeta, ComparisonMeta,
    _normalize_raw, _normalize_external,
)
from benchmarks.comprehensive.reporting.report_model import (
    Report, Chapter, Section, Block, TextBlock, TableBlock,
    FigureBlock, CalloutBlock, CommentaryBlock,
    MarkdownRenderer, HtmlRenderer, JsonManifestRenderer,
)
from benchmarks.comprehensive.reporting import lint


# ===================================================================
# 1. Result-store loading and normalization
# ===================================================================
class TestResultStoreLoading:
    def test_load_raw_dir(self):
        store = ResultStore()
        count = store.load_raw_dir(FIXTURES)
        assert count >= 6  # we have at least 6 fixture JSONs

    def test_normalized_fields_compression(self):
        data = _load_fixture("compression__pass__pbmc3k.json")
        row = _normalize_raw(data, source_path="fixtures/comp.json")
        assert row.benchmark == "compression"
        assert row.format == "scx_auto"
        assert row.dataset == "pbmc3k"
        assert row.file_size_bytes == 4720280
        assert row.median_wall_s is None
        assert row.source.kind == SourceKind.raw_json

    def test_normalized_fields_read_full(self):
        data = _load_fixture("read_full__pass__pbmc3k.json")
        row = _normalize_raw(data, source_path="fixtures/rf.json")
        assert row.benchmark == "read_full"
        assert row.median_wall_s == pytest.approx(0.123)
        assert len(row.runs) == 3

    def test_normalize_external(self):
        data = _load_fixture("harmony__external__census_1m.json")
        row = _normalize_external(data, source_path="fixtures/h.json")
        assert row.benchmark == "harmony_integrate"
        assert row.format == "scx_accel"
        assert row.median_wall_s == pytest.approx(42.5)
        assert row.source.kind == SourceKind.external_report

    def test_by_benchmark(self):
        store = ResultStore()
        store.load_raw_dir(FIXTURES)
        comp = store.by_benchmark("compression")
        assert len(comp) >= 1
        assert all(r.benchmark == "compression" for r in comp)

    def test_by_format(self):
        store = ResultStore()
        store.load_raw_dir(FIXTURES)
        scx = store.by_format("scx_auto")
        assert len(scx) >= 1

    def test_by_dataset(self):
        store = ResultStore()
        store.load_raw_dir(FIXTURES)
        pb = store.by_dataset("pbmc3k")
        assert len(pb) >= 1

    def test_latest_for(self):
        store = ResultStore()
        store.load_raw_dir(FIXTURES)
        row = store.latest_for("compression", "scx_auto", "pbmc3k")
        assert row is not None
        assert row.benchmark == "compression"

    def test_matrix(self):
        store = ResultStore()
        store.load_raw_dir(FIXTURES)
        m = store.matrix("compression", ["scx_auto"], ["pbmc3k"], "file_size_bytes")
        assert "scx_auto" in m
        assert m["scx_auto"]["pbmc3k"] == 4720280

    def test_coverage(self):
        store = ResultStore()
        store.load_raw_dir(FIXTURES)
        cov = store.coverage(["compression"], ["scx_auto", "zarr"], ["pbmc3k"])
        assert cov["compression"]["scx_auto"]["pbmc3k"] is True
        assert cov["compression"]["zarr"]["pbmc3k"] is False

    def test_add_manual(self):
        store = ResultStore()
        row = store.add_manual("compression", "manual_fmt", "ds1",
                               reason="Hand-measured", file_size_bytes=999)
        assert row.source.kind == SourceKind.manual
        assert row.source.reason == "Hand-measured"
        assert store.external_sources() == [row]

    def test_get_nested(self):
        data = _load_fixture("compression__pass__pbmc3k.json")
        row = _normalize_raw(data, source_path="x")
        assert row.get_nested("metadata.compression_ratio") == pytest.approx(4.12)
        assert row.get_nested("metadata.nonexistent") is None


# ===================================================================
# 2. Typed missing reasons
# ===================================================================
class TestMissingReasons:
    def test_enum_values(self):
        assert MissingReason.not_applicable == "not_applicable"
        assert MissingReason.not_supported == "not_supported"
        assert MissingReason.skipped_too_expensive == "skipped_too_expensive"
        assert MissingReason.missing_fixture == "missing_fixture"
        assert MissingReason.missing_dependency == "missing_dependency"
        assert MissingReason.benchmark_failed == "benchmark_failed"
        assert MissingReason.not_run == "not_run"
        assert MissingReason.not_preserved_by_converter == "not_preserved_by_converter"

    def test_all_members(self):
        assert len(MissingReason) == 8


# ===================================================================
# 3. Scenario and comparison metadata
# ===================================================================
class TestScenarioMeta:
    def test_from_raw_with_scenario(self):
        data = _load_fixture("read_full__pass__pbmc3k.json")
        sm = ScenarioMeta.from_raw(data)
        assert sm.name == "full_read"
        assert sm.mode == "full_pipeline"
        assert sm.thread_count == 8
        assert sm.cache_state == "warm"
        assert sm.device == "cpu"
        assert sm.storage_backend == "local"

    def test_from_raw_without_scenario(self):
        data = _load_fixture("compression__pass__pbmc3k.json")
        sm = ScenarioMeta.from_raw(data)
        assert sm.name is None
        assert sm.mode is None

    def test_batch_and_hvgs(self):
        data = _load_fixture("ml_loader__unsupported__pbmc3k.json")
        sm = ScenarioMeta.from_raw(data)
        assert sm.batch_size == 256
        assert sm.n_hvgs == 2000
        assert sm.device == "cpu"


class TestComparisonMeta:
    def test_from_raw_with_comparison(self):
        data = _load_fixture("correctness__fail__pbmc3k.json")
        cm = ComparisonMeta.from_raw(data)
        assert cm is not None
        assert cm.subject_impl == "scx_auto"
        assert cm.baseline_impl == "scanpy"
        assert cm.baseline_version == "1.12.0"
        assert cm.status == "fail"
        assert cm.value == pytest.approx(0.80)

    def test_from_raw_with_harness(self):
        data = _load_fixture("correctness__pass__pbmc3k.json")
        cm = ComparisonMeta.from_raw(data)
        assert cm is not None
        assert cm.baseline_impl == "scanpy"
        assert cm.status == "pass"

    def test_from_raw_none(self):
        data = _load_fixture("compression__pass__pbmc3k.json")
        cm = ComparisonMeta.from_raw(data)
        assert cm is None


# ===================================================================
# 4. Report model rendering — Markdown
# ===================================================================
def _minimal_report():
    """Build a small report for rendering tests."""
    return Report(
        title="Test Report",
        chapters=[
            Chapter(title="Summary", sections=[
                Section(title="Overview", blocks=[
                    TextBlock(content="This is a test report."),
                    TableBlock(
                        headers=["Format", "Size"],
                        rows=[["scx", "4.5 MB"], ["h5ad", "19.5 MB"]],
                        caption="File sizes",
                        source=SourceRef(kind=SourceKind.raw_json, path="comp.json"),
                    ),
                ]),
            ]),
            Chapter(title="Details", sections=[
                Section(title="Figures", blocks=[
                    FigureBlock(path="plots/test.png", caption="Test figure"),
                ]),
                Section(title="Notes", blocks=[
                    CalloutBlock(content="Important finding", level="warning"),
                    CommentaryBlock(
                        content="SCX achieves 4.2x compression",
                        source=SourceRef(kind=SourceKind.raw_json, path="comp.json"),
                    ),
                ]),
            ]),
        ],
    )


class TestMarkdownRenderer:
    def test_renders_title(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert "# Test Report" in md

    def test_renders_chapters(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert "## 1. Summary" in md
        assert "## 2. Details" in md

    def test_renders_sections(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert "### Overview" in md
        assert "### Figures" in md

    def test_renders_table(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert "| Format | Size |" in md
        assert "| scx | 4.5 MB |" in md
        assert "*File sizes*" in md

    def test_renders_figure(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert "![Test figure](plots/test.png)" in md

    def test_renders_callout(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert "Important finding" in md

    def test_renders_commentary_with_source(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert "4.2x compression" in md
        assert "_Source: raw_json" in md

    def test_renders_anchors(self):
        md = MarkdownRenderer().render(_minimal_report())
        assert '<a id="summary">' in md


# ===================================================================
# 5. Report model rendering — HTML
# ===================================================================
class TestHtmlRenderer:
    def test_renders_semantic_table(self):
        h = HtmlRenderer().render(_minimal_report())
        assert "<thead>" in h
        assert "<tbody>" in h
        assert "<th>" in h
        assert "<td>" in h

    def test_renders_caption(self):
        h = HtmlRenderer().render(_minimal_report())
        assert "<caption>" in h
        assert "File sizes" in h

    def test_renders_toc(self):
        h = HtmlRenderer().render(_minimal_report())
        assert '<nav class="toc">' in h

    def test_renders_callout(self):
        h = HtmlRenderer().render(_minimal_report())
        assert "callout-warning" in h

    def test_renders_figure(self):
        h = HtmlRenderer().render(_minimal_report())
        assert "<figure>" in h
        assert "plots/test.png" in h

    def test_source_ref_in_html(self):
        h = HtmlRenderer().render(_minimal_report())
        assert "source-ref" in h


# ===================================================================
# 6. Report model rendering — JSON
# ===================================================================
class TestJsonManifestRenderer:
    def test_valid_json(self):
        j = JsonManifestRenderer().render(_minimal_report())
        data = json.loads(j)
        assert data["title"] == "Test Report"

    def test_chapters_structure(self):
        data = json.loads(JsonManifestRenderer().render(_minimal_report()))
        assert len(data["chapters"]) == 2
        assert data["chapters"][0]["title"] == "Summary"
        assert data["chapters"][0]["number"] == 1

    def test_table_block_in_json(self):
        data = json.loads(JsonManifestRenderer().render(_minimal_report()))
        ch0 = data["chapters"][0]
        sec0 = ch0["sections"][0]
        table_block = [b for b in sec0["blocks"] if b["type"] == "table"][0]
        assert table_block["headers"] == ["Format", "Size"]
        assert table_block["rows"] == [["scx", "4.5 MB"], ["h5ad", "19.5 MB"]]
        assert table_block["source"]["kind"] == "raw_json"

    def test_figure_block_in_json(self):
        data = json.loads(JsonManifestRenderer().render(_minimal_report()))
        ch1 = data["chapters"][1]
        sec0 = ch1["sections"][0]
        fig = [b for b in sec0["blocks"] if b["type"] == "figure"][0]
        assert fig["path"] == "plots/test.png"


# ===================================================================
# 7. Lint — duplicate headings
# ===================================================================
class TestLintDuplicateHeadings:
    def test_no_duplicates(self):
        r = _minimal_report()
        warns = lint.check_duplicate_headings(r)
        assert len(warns) == 0

    def test_duplicate_chapter_titles(self):
        r = Report(title="T", chapters=[
            Chapter(title="Dup"), Chapter(title="Dup"),
        ])
        warns = lint.check_duplicate_headings(r)
        assert any(w.check == "duplicate_heading" for w in warns)

    def test_duplicate_anchor_ids(self):
        r = Report(title="T", chapters=[
            Chapter(title="A", id="same"),
            Chapter(title="B", id="same"),
        ])
        warns = lint.check_duplicate_headings(r)
        assert any(w.check == "duplicate_anchor" for w in warns)


# ===================================================================
# 8. Lint — missing source references
# ===================================================================
class TestLintMissingSources:
    def test_table_without_source(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(headers=["A"], rows=[["1"]], source=None),
                ]),
            ]),
        ])
        warns = lint.check_tables_have_sources(r)
        assert len(warns) == 1
        assert warns[0].check == "table_source_ref"

    def test_table_with_source(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(headers=["A"], rows=[["1"]],
                               source=SourceRef(kind=SourceKind.raw_json)),
                ]),
            ]),
        ])
        warns = lint.check_tables_have_sources(r)
        assert len(warns) == 0


# ===================================================================
# 9. Lint — missing figures
# ===================================================================
class TestLintMissingFigures:
    def test_missing_figure(self, tmp_path):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    FigureBlock(path="nonexistent.png"),
                ]),
            ]),
        ])
        warns = lint.check_missing_figures(r, figures_root=tmp_path)
        assert len(warns) == 1
        assert warns[0].check == "missing_figure"

    def test_existing_figure(self, tmp_path):
        (tmp_path / "ok.png").write_bytes(b"PNG")
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    FigureBlock(path="ok.png"),
                ]),
            ]),
        ])
        warns = lint.check_missing_figures(r, figures_root=tmp_path)
        assert len(warns) == 0


# ===================================================================
# 10. Lint — executive summary consistency
# ===================================================================
class TestLintExecConsistency:
    def test_matching_counts(self):
        r = Report(title="T", chapters=[
            Chapter(title="Executive Summary", sections=[
                Section(title="Overview", blocks=[
                    CalloutBlock(content="5/5 scanpy equivalence tests pass"),
                ]),
            ]),
            Chapter(title="Correctness", sections=[
                Section(title="Dataset summary", blocks=[
                    TableBlock(
                        headers=["Dataset", "Pass", "Fail", "Skip"],
                        rows=[["pbmc3k", "5", "0", "0"]],
                        caption="Dataset-level correctness",
                    ),
                ]),
            ]),
        ])
        warns = lint.check_executive_summary_consistency(r)
        assert len(warns) == 0

    def test_mismatched_counts(self):
        r = Report(title="T", chapters=[
            Chapter(title="Executive Summary", sections=[
                Section(title="Overview", blocks=[
                    CalloutBlock(content="10/10 scanpy equivalence tests pass"),
                ]),
            ]),
            Chapter(title="Correctness", sections=[
                Section(title="Dataset summary", blocks=[
                    TableBlock(
                        headers=["Dataset", "Pass", "Fail", "Skip"],
                        rows=[["pbmc3k", "5", "2", "1"]],
                        caption="Dataset-level correctness",
                    ),
                ]),
            ]),
        ])
        warns = lint.check_executive_summary_consistency(r)
        assert any(w.check == "executive_consistency" for w in warns)


# ===================================================================
# 11. Lint — HTML semantic tables
# ===================================================================
class TestLintHtmlSemantic:
    def test_semantic_elements(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(
                        headers=["A", "B"], rows=[["1", "2"]],
                        caption="Test",
                    ),
                ]),
            ]),
        ])
        warns = lint.check_html_semantic_tables(r)
        assert len(warns) == 0


# ===================================================================
# 12. Lint — public profile
# ===================================================================
class TestLintPublicProfile:
    def test_phase_labels_blocked(self):
        r = Report(title="T", chapters=[
            Chapter(title="Phase 5 results"),
        ])
        warns = lint.check_public_profile(r, public=True)
        assert any(w.check == "public_profile" for w in warns)

    def test_no_warning_when_not_public(self):
        r = Report(title="T", chapters=[
            Chapter(title="Phase 5 results"),
        ])
        warns = lint.check_public_profile(r, public=False)
        assert len(warns) == 0


# ===================================================================
# 13. Lint — skipped vs failed
# ===================================================================
class TestLintSkippedVsFailed:
    def test_fail_without_notes(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(
                        headers=["Test", "Status"],
                        rows=[["pca", "Pass"], ["umap", "Fail"]],
                    ),
                ]),
            ]),
        ])
        warns = lint.check_skipped_vs_failed(r)
        assert any(w.check == "skipped_vs_failed" for w in warns)

    def test_fail_with_notes(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(
                        headers=["Test", "Status", "Notes"],
                        rows=[["umap", "Fail", "tolerance exceeded"]],
                    ),
                ]),
            ]),
        ])
        warns = lint.check_skipped_vs_failed(r)
        assert len(warns) == 0


# ===================================================================
# 14. Lint — empty cells
# ===================================================================
class TestLintEmptyCells:
    def test_mostly_empty_row(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(
                        headers=["A", "B", "C", "D"],
                        rows=[["x", "—", "—", "—"]],
                    ),
                ]),
            ]),
        ])
        warns = lint.check_empty_cells(r)
        assert any(w.check == "empty_cells" for w in warns)


# ===================================================================
# 15. Lint — approximate numbers
# ===================================================================
class TestLintApproxNumbers:
    def test_tilde_number(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TextBlock(content="Achieves ~4.2x compression ratio"),
                ]),
            ]),
        ])
        warns = lint.check_approximate_number_sources(r)
        assert any(w.check == "approximate_number_source" for w in warns)


# ===================================================================
# 16. Lint — memory mode consistency
# ===================================================================
class TestLintMemoryMode:
    def test_mismatch(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(
                        headers=["Fmt", "RSS"],
                        rows=[["scx", "45"]],
                        caption="Full read memory",
                    ),
                    TextBlock(content="In backed mode, memory is lower."),
                ]),
            ]),
        ])
        warns = lint.check_memory_mode_consistency(r)
        assert any(w.check == "memory_mode_consistency" for w in warns)


# ===================================================================
# 17. Golden Markdown snapshot
# ===================================================================
class TestGoldenMarkdown:
    def test_golden_snapshot(self):
        md = MarkdownRenderer().render(_minimal_report())
        # Verify structural invariants of the golden output
        lines = md.split("\n")
        assert lines[0] == "# Test Report"
        assert any("## 1. Summary" in l for l in lines)
        assert any("## 2. Details" in l for l in lines)
        assert any("| Format | Size |" in l for l in lines)
        assert any("| scx | 4.5 MB |" in l for l in lines)
        assert any("*File sizes*" in l for l in lines)
        assert md.endswith("\n")


# ===================================================================
# 18. Golden HTML snapshot — semantic tables
# ===================================================================
class TestGoldenHtml:
    def test_semantic_table_structure(self):
        h = HtmlRenderer().render(_minimal_report())
        assert "<!DOCTYPE html>" in h
        assert "<thead>" in h
        assert "<tbody>" in h
        assert "<caption>File sizes</caption>" in h
        assert "<th>Format</th>" in h
        assert "<td>scx</td>" in h

    def test_wide_table_wrapper(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(headers=["A"], rows=[["1"]], wide=True),
                ]),
            ]),
        ])
        h = HtmlRenderer().render(r)
        assert 'class="table-wide"' in h


# ===================================================================
# 19. Section numbering stability
# ===================================================================
class TestSectionNumbering:
    def test_chapter_numbers_sequential(self):
        r = Report(title="T", chapters=[
            Chapter(title="A"),
            Chapter(title="B"),
            Chapter(title="C"),
        ])
        r.generate_ids_and_numbers()
        assert [c.number for c in r.chapters] == [1, 2, 3]

    def test_ids_stable_across_calls(self):
        r = _minimal_report()
        r.generate_ids_and_numbers()
        ids1 = [(c.id, [s.id for s in c.sections]) for c in r.chapters]
        r.generate_ids_and_numbers()
        ids2 = [(c.id, [s.id for s in c.sections]) for c in r.chapters]
        assert ids1 == ids2

    def test_custom_ids_preserved(self):
        r = Report(title="T", chapters=[
            Chapter(title="A", id="custom-a", sections=[
                Section(title="S", id="custom-s"),
            ]),
        ])
        r.generate_ids_and_numbers()
        assert r.chapters[0].id == "custom-a"
        assert r.chapters[0].sections[0].id == "custom-s"

    def test_subsection_ids(self):
        r = Report(title="T", chapters=[
            Chapter(title="Ch", sections=[
                Section(title="Sec", subsections=[
                    Section(title="Sub"),
                ]),
            ]),
        ])
        r.generate_ids_and_numbers()
        sub = r.chapters[0].sections[0].subsections[0]
        assert sub.id is not None
        assert "sub" in sub.id


# ===================================================================
# 20. Wide-table appendix routing
# ===================================================================
class TestWideTableRouting:
    def test_wide_flag(self):
        t = TableBlock(headers=["A", "B", "C"], rows=[["1", "2", "3"]], wide=True)
        assert t.wide is True

    def test_wide_in_markdown(self):
        # Wide tables render normally in markdown (no special wrapper)
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(headers=["A"], rows=[["1"]], wide=True),
                ]),
            ]),
        ])
        md = MarkdownRenderer().render(r)
        assert "| A |" in md

    def test_wide_in_html(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(headers=["A"], rows=[["1"]], wide=True),
                ]),
            ]),
        ])
        h = HtmlRenderer().render(r)
        assert 'table-wide' in h


# ===================================================================
# 21. Lint — collect_warnings aggregation
# ===================================================================
class TestLintCollectWarnings:
    def test_returns_sorted(self):
        r = Report(title="T", chapters=[
            Chapter(title="Dup"), Chapter(title="Dup"),
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    TableBlock(headers=["A"], rows=[["1"]]),
                ]),
            ]),
        ])
        warns = lint.collect_warnings(r, figures_root=Path("/tmp"))
        # errors should come before info
        levels = [w.level for w in warns]
        error_idx = [i for i, l in enumerate(levels) if l == lint.LintLevel.error]
        info_idx = [i for i, l in enumerate(levels) if l == lint.LintLevel.info]
        if error_idx and info_idx:
            assert max(error_idx) < min(info_idx)

    def test_strict_promotes_manual(self):
        r = Report(title="T", chapters=[
            Chapter(title="C", sections=[
                Section(title="S", blocks=[
                    CommentaryBlock(
                        content="Achieves 4.2x compression",
                        source=SourceRef(kind=SourceKind.manual, reason="test"),
                    ),
                ]),
            ]),
        ])
        warns = lint.collect_warnings(r, strict=True, figures_root=Path("/tmp"))
        manual = [w for w in warns if w.check == "manual_numeric_claims"]
        assert all(w.level == lint.LintLevel.error for w in manual)


# ===================================================================
# 22. SourceRef and SourceKind
# ===================================================================
class TestSourceRef:
    def test_repr(self):
        sr = SourceRef(kind=SourceKind.raw_json, path="foo.json")
        assert "raw_json" in repr(sr)
        assert "foo.json" in repr(sr)

    def test_manual_requires_reason(self):
        sr = SourceRef(kind=SourceKind.manual, reason="hand-measured")
        assert sr.reason == "hand-measured"

    def test_all_kinds(self):
        assert len(SourceKind) == 5
