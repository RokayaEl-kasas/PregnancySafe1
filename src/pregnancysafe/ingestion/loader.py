"""Document loading and text cleaning — PDFs and HTML pages.

Reads every PDF and HTML file found recursively under
data/raw/<disease_id>/ and converts them into clean text
before chunking.

The loader is intentionally responsible only for:
    1. Reading PDF / HTML files.
    2. Cleaning extraction artifacts.
    3. Resolving source names and URLs.

Chunking is handled separately by ingestion/chunker.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pregnancysafe.ingestion.manifest import lookup_source_url
from pregnancysafe.schemas import Disease
from pregnancysafe.utils.logging_config import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text-cleaning patterns
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# PDF extraction often produces:
#     preg-
#     nancy
#
# This restores:
#     pregnancy
#
# \w is intentionally used here because most medical source text is English.
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# Remove spaces that appear immediately after a line break.
_LINEBREAK_SPACE_RE = re.compile(r"\n[ \t]+")

# PDF extraction sometimes leaves spaces before punctuation.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

# Normalize repeated spaces around line breaks.
_MULTI_SPACE_NEWLINE_RE = re.compile(r"[ \t]+\n[ \t]+")


def _clean_text(raw_text: str) -> str:
    """Clean common PDF/HTML extraction artifacts.

    The cleaning is deliberately conservative.

    We do NOT aggressively join every line because PDFs often contain
    headings, bullet points, tables, and separate sections where line
    boundaries carry useful information.

    Main operations:
        - Remove null characters.
        - Restore hyphenated words split across PDF lines.
        - Normalize spaces.
        - Remove excessive blank lines.
        - Clean spaces around punctuation.
    """

    if not raw_text:
        return ""

    text = raw_text.replace("\x00", "")

    # ---------------------------------------------------------------
    # 1. Restore words split by PDF line wrapping.
    # ---------------------------------------------------------------
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # ---------------------------------------------------------------
    # 2. Normalize tabs / horizontal whitespace.
    # ---------------------------------------------------------------
    text = _WHITESPACE_RE.sub(" ", text)

    # ---------------------------------------------------------------
    # 3. Remove indentation accidentally introduced by extraction.
    # ---------------------------------------------------------------
    text = _LINEBREAK_SPACE_RE.sub("\n", text)

    text = _MULTI_SPACE_NEWLINE_RE.sub("\n", text)

    # ---------------------------------------------------------------
    # 4. Remove spaces before punctuation.
    # ---------------------------------------------------------------
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)

    # ---------------------------------------------------------------
    # 5. Collapse excessive blank lines.
    # ---------------------------------------------------------------
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def load_pdf_text(pdf_path: Path) -> str:
    """Extract and clean text from a single PDF file.

    Requires:
        pip install pypdf

    Each page is extracted independently so one problematic page
    does not stop the entire document from being ingested.
    """

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pypdf is required for PDF ingestion. "
            "Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))

    pages_text: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages_text.append(page_text)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to extract page %d of %s: %s",
                page_num,
                pdf_path.name,
                exc,
            )

    # Keep page boundaries separated.
    raw_text = "\n\n".join(pages_text)

    cleaned_text = _clean_text(raw_text)

    logger.debug(
        "Loaded PDF %s: %d raw chars -> %d cleaned chars",
        pdf_path.name,
        len(raw_text),
        len(cleaned_text),
    )

    return cleaned_text


# ---------------------------------------------------------------------------
# HTML loading
# ---------------------------------------------------------------------------

def load_html_text(html_path: Path) -> str:
    """Extract visible body text from a saved HTML page.

    Removes common website chrome such as:
        - script
        - style
        - navigation
        - header
        - footer
        - noscript

    Requires:
        pip install beautifulsoup4
    """

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "beautifulsoup4 is required for HTML ingestion. "
            "Install with: pip install beautifulsoup4"
        ) from exc

    with open(
        html_path,
        encoding="utf-8",
        errors="replace",
    ) as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content website elements.
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "noscript",
        ]
    ):
        tag.decompose()

    text = soup.get_text(separator="\n")

    return _clean_text(text)


# ---------------------------------------------------------------------------
# Loaded document model
# ---------------------------------------------------------------------------

@dataclass
class LoadedDocument:
    path: Path
    text: str
    source_name: str
    source_url: str


# ---------------------------------------------------------------------------
# Source-name handling
# ---------------------------------------------------------------------------

def _clean_source_title(value: str) -> str:
    """Convert a raw filename/URL component into a readable title."""

    value = value.strip()

    if not value:
        return ""

    # Remove common file extensions.
    value = re.sub(
        r"\.(pdf|html?|htm)$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # Replace separators with spaces.
    value = re.sub(r"[_\-]+", " ", value)

    # Remove obvious random hash suffixes.
    value = re.sub(
        r"\s+[a-f0-9]{8,}$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value.title()


def _source_name_from_url(source_url: str) -> str:
    """Try to derive a readable source name from a source URL.

    Examples:

        https://www.nice.org.uk/guidance/ng133/...
            -> NICE

        https://www.ncbi.nlm.nih.gov/books/...
            -> NCBI

        https://www.who.int/...
            -> WHO
    """

    if not source_url:
        return ""

    try:
        parsed = urlparse(source_url)
        hostname = (parsed.hostname or "").lower()

    except Exception:
        return ""

    # ---------------------------------------------------------------
    # Known official medical sources.
    # ---------------------------------------------------------------

    if "nice.org.uk" in hostname:
        return "NICE"

    if "who.int" in hostname:
        return "WHO"

    if "ncbi.nlm.nih.gov" in hostname:
        return "NCBI"

    if "cdc.gov" in hostname:
        return "CDC"

    if "nih.gov" in hostname:
        return "NIH"

    if "fda.gov" in hostname:
        return "FDA"

    if "nhlbi.nih.gov" in hostname:
        return "NHLBI"

    # ---------------------------------------------------------------
    # Generic fallback.
    # ---------------------------------------------------------------

    hostname = hostname.removeprefix("www.")

    if hostname:
        first_part = hostname.split(".")[0]
        return first_part.upper()

    return ""


def _derive_source_name(
    file_path: Path,
    disease_folder: Path,
    fallback: str,
    source_url: Optional[str] = None,
) -> str:
    """Derive a human-readable source name.

    Priority:

        1. Medication folder name.
        2. Recognized source from URL.
        3. Filename when it is descriptive.
        4. Configured fallback.

    Examples:

        data/raw/hypertension/
            medications/
                labetalol/
                    something.pdf

    becomes:

        Labetalol

    A NICE URL becomes:

        NICE

    A WHO URL becomes:

        WHO
    """

    try:
        rel_parts = file_path.relative_to(disease_folder).parts

    except ValueError:
        rel_parts = ()

    # ---------------------------------------------------------------
    # Medication source.
    #
    # medications / labetalol / file.pdf
    # ---------------------------------------------------------------

    if (
        len(rel_parts) >= 3
        and rel_parts[0].lower() == "medications"
    ):
        return _clean_source_title(rel_parts[1])

    # ---------------------------------------------------------------
    # Try the URL before using the disease fallback.
    # ---------------------------------------------------------------

    url_source = _source_name_from_url(source_url or "")

    if url_source:
        return url_source

    # ---------------------------------------------------------------
    # Try to derive a name from the filename.
    #
    # We only use this for guideline/source files where the filename
    # contains meaningful text.
    # ---------------------------------------------------------------

    stem = file_path.stem

    cleaned_stem = _clean_source_title(stem)

    # Avoid treating completely generic generated filenames as source
    # names, e.g. pdf_56fa7572d5.
    if cleaned_stem:
        generic_patterns = [
            r"^pdf\s+[a-f0-9]{6,}$",
            r"^[a-f0-9]{16,}$",
            r"^document\s+\d+$",
        ]

        if not any(
            re.match(pattern, cleaned_stem, flags=re.IGNORECASE)
            for pattern in generic_patterns
        ):
            return cleaned_stem

    return fallback


# ---------------------------------------------------------------------------
# Disease document loading
# ---------------------------------------------------------------------------

def load_disease_documents(
    disease: Disease,
    raw_data_root: Path,
    manifest: Optional[dict[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> list[LoadedDocument]:
    """Recursively load every PDF/HTML file under a disease folder.

    Expected structure:

        data/raw/<disease_id>/
            guidelines/
                *.pdf
                *.html

            medications/
                <medication>/
                    *.pdf
                    *.html

    Source URL resolution priority:

        1. Exact/suffix match in download manifest.
        2. Disease's first verified_open source in config.yaml.

    Files with no extractable text are skipped.
    """

    folder = raw_data_root / disease.id

    if not folder.exists():
        logger.warning(
            "Raw data folder does not exist yet: %s",
            folder,
        )
        return []

    manifest = manifest or {}

    repo_root = (
        repo_root
        or raw_data_root.parent.parent
    )

    ingestible_sources = disease.ingestible_sources()

    fallback_source = (
        ingestible_sources[0]
        if ingestible_sources
        else None
    )

    fallback_name = (
        fallback_source.name
        if fallback_source
        else disease.label_en
    )

    # ---------------------------------------------------------------
    # Find all supported files recursively.
    # ---------------------------------------------------------------

    files = (
        sorted(folder.rglob("*.pdf"))
        + sorted(folder.rglob("*.html"))
        + sorted(folder.rglob("*.htm"))
    )

    if not files:
        logger.info(
            "No PDF/HTML files found under %s — "
            "download sources first.",
            folder,
        )
        return []

    documents: list[LoadedDocument] = []

    # ---------------------------------------------------------------
    # Process each document.
    # ---------------------------------------------------------------

    for file_path in files:

        try:
            if file_path.suffix.lower() == ".pdf":
                text = load_pdf_text(file_path)

            else:
                text = load_html_text(file_path)

        except ImportError:
            # Dependency problems should stop ingestion clearly.
            raise

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s: %s",
                file_path,
                exc,
            )
            continue

        # -----------------------------------------------------------
        # Skip empty documents.
        # -----------------------------------------------------------

        if not text.strip():
            logger.info(
                "Skipping %s — no extractable text "
                "(likely a scanned PDF or landing page).",
                file_path,
            )
            continue

        # -----------------------------------------------------------
        # Resolve the real source URL.
        # -----------------------------------------------------------

        source_url = lookup_source_url(
            manifest,
            file_path,
            repo_root,
        )

        # -----------------------------------------------------------
        # Resolve source name.
        #
        # IMPORTANT:
        # source_name is now resolved using the actual source URL
        # instead of automatically using the first disease source.
        # -----------------------------------------------------------

        source_name = _derive_source_name(
            file_path=file_path,
            disease_folder=folder,
            fallback=fallback_name,
            source_url=source_url,
        )

        # -----------------------------------------------------------
        # If URL is missing, use the configured fallback source.
        # -----------------------------------------------------------

        if not source_url:

            if fallback_source:
                source_url = fallback_source.url

                # If the filename/path didn't give us a useful source
                # name, use the configured fallback.
                if not source_name:
                    source_name = fallback_source.name

            else:
                logger.info(
                    "Skipping %s — no manifest entry and no "
                    "fallback source configured.",
                    file_path,
                )
                continue

        # -----------------------------------------------------------
        # Final safety fallback.
        # -----------------------------------------------------------

        if not source_name:
            source_name = disease.label_en

        # -----------------------------------------------------------
        # Store document.
        # -----------------------------------------------------------

        documents.append(
            LoadedDocument(
                path=file_path,
                text=text,
                source_name=source_name,
                source_url=source_url,
            )
        )

        logger.debug(
            "Loaded document: %s | source=%s | url=%s | chars=%d",
            file_path.name,
            source_name,
            source_url,
            len(text),
        )

    logger.info(
        "Loaded %d document(s) for disease=%s",
        len(documents),
        disease.id,
    )

    return documents