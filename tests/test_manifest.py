"""Tests for ingestion.manifest (download-log parsing) and the recursive,
multi-format loader.load_disease_documents — the pieces that make ingestion
work against a real batch-downloader's nested folder + Windows-path log,
like the one the user's own downloader script produces.
"""

from __future__ import annotations

from pathlib import Path

from pregnancysafe.ingestion.loader import load_disease_documents, load_html_text
from pregnancysafe.ingestion.manifest import load_download_manifest, lookup_source_url
from pregnancysafe.schemas import Disease, DocumentSource, SourceStatus

SAMPLE_LOG = r"""PregnancySafe Source Download Log
============================================================

Successful: 3
Failed:     1
Skipped:    1

SUCCESSFUL
------------------------------------------------------------
DOWNLOADED: data\raw\uti\medications\cephalexin\pdf.pdf <- https://www.ncbi.nlm.nih.gov/books/NBK605062/pdf/
DOWNLOADED: data\raw\uti\guidelines\www_acog_org_668eaddcf9.html <- https://www.acog.org/clinical/clinical-guidance/clinical-consensus/articles/2023/08/urinary-tract-infections-in-pregnant-individuals
EXISTS: data\raw\uti\medications\cephalexin\pdf.pdf

FAILED
------------------------------------------------------------
FAILED: https://pubmed.ncbi.nlm.nih.gov/31302868/ | HTTP 203

SKIPPED
------------------------------------------------------------
UNCLASSIFIED: https://www.ncbi.nlm.nih.gov/books/NBK582980/ | section=whatever
"""


class TestManifestParsing:
    def test_parses_downloaded_lines(self, tmp_path: Path):
        log_path = tmp_path / "download_sources.log"
        log_path.write_text(SAMPLE_LOG, encoding="utf-8")

        manifest = load_download_manifest(log_path)
        assert "data/raw/uti/medications/cephalexin/pdf.pdf" in manifest
        assert manifest["data/raw/uti/medications/cephalexin/pdf.pdf"] == (
            "https://www.ncbi.nlm.nih.gov/books/NBK605062/pdf/"
        )

    def test_ignores_exists_failed_and_skipped_lines(self, tmp_path: Path):
        log_path = tmp_path / "download_sources.log"
        log_path.write_text(SAMPLE_LOG, encoding="utf-8")

        manifest = load_download_manifest(log_path)
        # Only 2 DOWNLOADED lines in the sample -> exactly 2 entries
        assert len(manifest) == 2

    def test_missing_log_returns_empty_dict(self, tmp_path: Path):
        assert load_download_manifest(tmp_path / "does_not_exist.log") == {}

    def test_windows_backslashes_normalized(self, tmp_path: Path):
        log_path = tmp_path / "download_sources.log"
        log_path.write_text(SAMPLE_LOG, encoding="utf-8")
        manifest = load_download_manifest(log_path)
        assert all("\\" not in key for key in manifest)


class TestLookupSourceUrl:
    def test_exact_relative_match(self, tmp_path: Path):
        manifest = {"data/raw/uti/medications/cephalexin/pdf.pdf": "https://example.org/cephalexin"}
        repo_root = tmp_path
        file_path = tmp_path / "data/raw/uti/medications/cephalexin/pdf.pdf"
        assert lookup_source_url(manifest, file_path, repo_root) == "https://example.org/cephalexin"

    def test_no_match_returns_none(self, tmp_path: Path):
        manifest = {"data/raw/uti/medications/cephalexin/pdf.pdf": "https://example.org/cephalexin"}
        file_path = tmp_path / "data/raw/anemia/medications/iron/pdf.pdf"
        assert lookup_source_url(manifest, file_path, tmp_path) is None


class TestHtmlLoading:
    def test_extracts_visible_text_and_strips_scripts(self, tmp_path: Path):
        html_path = tmp_path / "page.html"
        html_path.write_text(
            "<html><body><script>evil()</script>"
            "<h1>UTI Consensus</h1><p>Nitrofurantoin is first-line therapy.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        text = load_html_text(html_path)
        assert "Nitrofurantoin is first-line therapy" in text
        assert "evil()" not in text


class TestLoadDiseaseDocumentsRecursive:
    def _make_disease(self, folder: str) -> Disease:
        return Disease(
            id="uti",
            label_ar="التهابات المسالك البولية",
            label_en="UTI",
            folder=folder,
            sources=[
                DocumentSource(
                    name="ACOG Fallback",
                    url="https://acog.org/fallback",
                    status=SourceStatus.VERIFIED_OPEN,
                )
            ],
        )

    def test_finds_nested_html_and_uses_manifest_citation(self, tmp_path: Path):
        raw_root = tmp_path / "data" / "raw"
        html_path = raw_root / "uti" / "guidelines" / "www_acog_org_668eaddcf9.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            "<html><body><p>Cephalexin is a first-line alternative for UTI in pregnancy.</p></body></html>",
            encoding="utf-8",
        )

        manifest = {
            "data/raw/uti/guidelines/www_acog_org_668eaddcf9.html": "https://www.acog.org/real-uti-page"
        }

        disease = self._make_disease(folder=str(raw_root / "uti"))
        docs = load_disease_documents(disease, raw_root, manifest=manifest, repo_root=tmp_path)

        assert len(docs) == 1
        assert docs[0].source_url == "https://www.acog.org/real-uti-page"
        assert "Cephalexin" in docs[0].text

    def test_falls_back_to_disease_default_source_without_manifest(self, tmp_path: Path):
        raw_root = tmp_path / "data" / "raw"
        html_path = raw_root / "uti" / "guidelines" / "some_page.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text("<html><body><p>Some UTI content.</p></body></html>", encoding="utf-8")

        disease = self._make_disease(folder=str(raw_root / "uti"))
        docs = load_disease_documents(disease, raw_root, manifest={}, repo_root=tmp_path)

        assert len(docs) == 1
        assert docs[0].source_url == "https://acog.org/fallback"

    def test_missing_folder_returns_empty_list(self, tmp_path: Path):
        disease = self._make_disease(folder=str(tmp_path / "data" / "raw" / "uti"))
        docs = load_disease_documents(disease, tmp_path / "data" / "raw", manifest={}, repo_root=tmp_path)
        assert docs == []

    def test_empty_extracted_text_is_skipped(self, tmp_path: Path):
        raw_root = tmp_path / "data" / "raw"
        html_path = raw_root / "uti" / "guidelines" / "blank.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text("<html><body></body></html>", encoding="utf-8")

        disease = self._make_disease(folder=str(raw_root / "uti"))
        docs = load_disease_documents(disease, raw_root, manifest={}, repo_root=tmp_path)
        assert docs == []
