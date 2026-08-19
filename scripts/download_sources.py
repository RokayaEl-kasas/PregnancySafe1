#!/usr/bin/env python3
"""
PregnancySafe - Source Downloader

Reads:
    docs/PregnancySafe_Medication_Links.md
    docs/PregnancySafe_Medical_Sources.md

Downloads sources into:
    data/raw/<disease>/
        guidelines/
        medications/<medication>/

Creates:
    data/raw/download_sources.log

The downloader uses explicit disease/medication mappings to avoid
misclassifying numbered Markdown sections.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
RAW_DIR = REPO_ROOT / "data" / "raw"
LOG_PATH = RAW_DIR / "download_sources.log"


# ============================================================
# CANONICAL DISEASE MAPPING
# ============================================================

DISEASE_ALIASES = {
    "nausea_vomiting": [
        "nausea",
        "vomiting",
        "hyperemesis",
        "nausea and vomiting",
    ],
    "gerd_heartburn": [
        "heartburn",
        "gerd",
        "reflux",
        "acid reflux",
    ],
    "hypertension": [
        "hypertension",
        "preeclampsia",
        "hypertensive",
    ],
    "uti": [
        "uti",
        "urinary tract infection",
        "urinary tract infections",
    ],
    "gestational_diabetes": [
        "gestational diabetes",
        "gdm",
    ],
    "anemia": [
        "anemia",
        "iron deficiency",
    ],
    "thyroid": [
        "thyroid",
        "hypothyroidism",
        "hyperthyroidism",
    ],
    "antenatal_care": [
        "antenatal care",
        "antenatal",
        "prenatal care",
    ],
    "varicose_veins": [
        "varicose",
        "leg edema",
        "leg oedema",
    ],
}


# ============================================================
# MEDICATION MAPPING
# ============================================================

MEDICATION_BY_URL = {
    "NBK582681": ("nausea_vomiting", "doxylamine_pyridoxine"),
    "NBK582840": ("nausea_vomiting", "metoclopramide"),
    "NBK582886": ("nausea_vomiting", "ondansetron"),
    "NBK582916": ("nausea_vomiting", "promethazine"),

    "NBK582699": ("gerd_heartburn", "famotidine"),
    "NBK582884": ("gerd_heartburn", "omeprazole_esomeprazole"),
    "NBK582920": ("gerd_heartburn", "proton_pump_inhibitors"),

    "NBK582779": ("hypertension", "labetalol"),
    "NBK582876": ("hypertension", "nifedipine"),
    "NBK501026": ("hypertension", "methyldopa"),
    "NBK582517": ("hypertension", "ace_inhibitors"),

    "NBK605076": ("uti", "nitrofurantoin"),
    "NBK501053": ("uti", "nitrofurantoin_g6pd"),
    "NBK605062": ("uti", "cephalexin"),

    "NBK582828": ("gestational_diabetes", "metformin"),
    "NBK605077": ("gestational_diabetes", "insulin_glargine"),
    "NBK605063": ("gestational_diabetes", "insulin_aspart"),
    "NBK500865": ("gestational_diabetes", "glyburide"),
    "NBK582729": ("gestational_diabetes", "insulin"),

    "NBK614541": ("anemia", "iron"),

    "NBK614540": ("thyroid", "methimazole"),
    "NBK582545": ("thyroid", "propylthiouracil"),
    "NBK501003": ("thyroid", "levothyroxine"),

    "31302868": ("uti", "uti_medication_reference"),

    # FDA NSAID pregnancy warning
    "fda.gov": ("hypertension", "nsaids_pregnancy_warning"),

    "thyroid.org": ("thyroid", "hypothyroidism_pregnancy"),
}


# ============================================================
# GUIDELINE MAPPING
# ============================================================

GUIDELINE_BY_URL = {
    "who.int/publications/i/item/9789241549912": "antenatal_care",
    "iris.who.int/server/api/core/bitstreams/9dccde13-3593-4a22-9237-61abe5a3c6b7": "antenatal_care",

    "nice.org.uk/guidance/ng201": "antenatal_care",

    "acog.org/clinical/clinical-guidance/practice-bulletin/articles/2018/01/nausea-and-vomiting-of-pregnancy":
        "nausea_vomiting",

    "ncbi.nlm.nih.gov/books/NBK140561": "hypertension",

    "ncbi.nlm.nih.gov/books/n/whoeclampsia": "hypertension",

    "nice.org.uk/guidance/ng133": "hypertension",

    "acog.org/clinical/clinical-guidance/practice-bulletin/articles/2021/08/anemia-in-pregnancy":
        "anemia",

    "pubmed.ncbi.nlm.nih.gov/34293770": "anemia",

    "acog.org/clinical/clinical-guidance/clinical-consensus/articles/2023/08/urinary-tract-infections-in-pregnant-individuals":
        "uti",

    "pubmed.ncbi.nlm.nih.gov/42219800": "thyroid",

    "cochrane.org/evidence/CD001066_interventions-varicose-veins-and-leg-oedema-pregnancy":
        "varicose_veins",

    "ncbi.nlm.nih.gov/books/NBK327998": "varicose_veins",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def url_key(url: str) -> str:
    """
    Creates a normalized lookup key from a URL.
    """
    url = normalize_url(url)
    parsed = urlparse(url)

    return f"{parsed.netloc}{parsed.path}".lower()


def safe_filename(url: str) -> str:
    """
    Generates a stable filename from the URL.
    """
    parsed = urlparse(url)

    name = Path(parsed.path).name

    if not name:
        name = parsed.netloc

    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

    if not name:
        name = "source"

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]

    return f"{name}_{digest}"


def find_disease_from_text(text: str) -> str | None:
    """
    Finds a canonical disease from surrounding Markdown text.
    """

    lower = text.lower()

    # Long/specific aliases first
    candidates = []

    for disease, aliases in DISEASE_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower:
                candidates.append((len(alias), disease))

    if not candidates:
        return None

    candidates.sort(reverse=True)

    return candidates[0][1]


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from Markdown.
    """
    urls = re.findall(
        r"https?://[^\s<>\]\)]+",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = []

    for url in urls:
        url = url.rstrip(".,;")

        if url not in cleaned:
            cleaned.append(url)

    return cleaned


def classify_url(url: str, context: str, source_type: str):
    """
    Classifies URL into canonical disease/medication.
    """

    normalized = url_key(url)

    # Medication URL mapping
    if source_type == "medication":

        for key, value in MEDICATION_BY_URL.items():

            if key.lower() in normalized:
                return value

        # Generic thyroid source
        if "thyroid.org" in normalized:
            return ("thyroid", "hypothyroidism_pregnancy")

        # FDA NSAID source
        if "fda.gov" in normalized:
            return ("hypertension", "nsaids_pregnancy_warning")

        # Fallback based on context
        disease = find_disease_from_text(context)

        if disease:
            return (disease, "unknown")

        return (None, None)

    # Guideline URL mapping
    if source_type == "guideline":

        for key, disease in GUIDELINE_BY_URL.items():

            if key.lower() in normalized:
                return (disease, None)

        disease = find_disease_from_text(context)

        if disease:
            return (disease, None)

    return (None, None)


def download_file(url: str, destination: Path) -> tuple[bool, str]:
    """
    Download a single URL.
    """

    try:

        destination.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(
            url,
            timeout=60,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/151.0 Safari/537.36"
                )
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        destination.write_bytes(response.content)

        return True, f"HTTP {response.status_code}"

    except Exception as exc:

        return False, str(exc)


# ============================================================
# READ MARKDOWN SOURCES
# ============================================================

def read_sources():

    medication_file = DOCS_DIR / "PregnancySafe_Medication_Links.md"
    guideline_file = DOCS_DIR / "PregnancySafe_Medical_Sources.md"

    sources = []

    for path, source_type in [
        (medication_file, "medication"),
        (guideline_file, "guideline"),
    ]:

        if not path.exists():
            print(f"WARNING: Missing {path}")
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        urls = extract_urls(text)

        for url in urls:

            # Grab nearby context around URL
            position = text.find(url)

            start = max(0, position - 1000)
            end = min(len(text), position + len(url) + 1000)

            context = text[start:end]

            sources.append(
                {
                    "url": url,
                    "type": source_type,
                    "context": context,
                }
            )

    # Deduplicate
    unique = {}

    for source in sources:
        unique[(source["url"], source["type"])] = source

    return list(unique.values())


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PregnancySafe - Source Downloader")
    print("=" * 60)

    print()
    print(f"Repository: {REPO_ROOT}")
    print(f"Docs:       {DOCS_DIR}")
    print(f"Raw data:   {RAW_DIR}")
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading source files...")

    sources = read_sources()

    medication_count = sum(
        1 for s in sources if s["type"] == "medication"
    )

    guideline_count = sum(
        1 for s in sources if s["type"] == "guideline"
    )

    print(f"Medication URLs: {medication_count}")
    print(f"Guideline URLs:  {guideline_count}")
    print(f"Unique URLs:     {len(sources)}")
    print()

    successful = []
    failed = []
    skipped = []

    for index, source in enumerate(sources, start=1):

        url = source["url"]
        source_type = source["type"]
        context = source["context"]

        disease, medication = classify_url(
            url,
            context,
            source_type,
        )

        print(f"[{index}/{len(sources)}]")
        print(f"Type:    {source_type}")
        print(f"URL:     {url}")

        if disease is None:

            print("  SKIPPED: Could not classify source")
            skipped.append(
                f"UNCLASSIFIED: {url}"
            )

            print()
            continue

        # ----------------------------------------------------
        # Medication
        # ----------------------------------------------------

        if source_type == "medication":

            medication = medication or "unknown"

            directory = (
                RAW_DIR
                / disease
                / "medications"
                / medication
            )

        # ----------------------------------------------------
        # Guideline
        # ----------------------------------------------------

        else:

            directory = (
                RAW_DIR
                / disease
                / "guidelines"
            )

        filename = safe_filename(url)

        # Determine extension
        lower_url = url.lower()

        if (
            "pdf" in lower_url
            or lower_url.endswith(".pdf")
        ):
            filename += ".pdf"

        else:
            filename += ".html"

        destination = directory / filename

        if destination.exists():

            print(
                f"  EXISTS: {destination.relative_to(REPO_ROOT)}"
            )

            successful.append(
                f"EXISTS: {destination.relative_to(REPO_ROOT)}"
            )

            print()
            continue

        ok, message = download_file(
            url,
            destination,
        )

        if ok:

            relative = destination.relative_to(REPO_ROOT)

            print(
                f"  DOWNLOADED: {relative} <- {url}"
            )

            successful.append(
                f"DOWNLOADED: {relative} <- {url}"
            )

        else:

            print(
                f"  FAILED: {url} | {message}"
            )

            failed.append(
                f"FAILED: {url} | {message}"
            )

        print()

    # ========================================================
    # LOG
    # ========================================================

    lines = [
        "PregnancySafe Source Download Log",
        "=" * 60,
        "",
        f"Successful: {len(successful)}",
        f"Failed:     {len(failed)}",
        f"Skipped:    {len(skipped)}",
        "",
        "SUCCESSFUL",
        "-" * 60,
        *successful,
        "",
        "FAILED",
        "-" * 60,
        *failed,
        "",
        "SKIPPED",
        "-" * 60,
        *skipped,
        "",
    ]

    LOG_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")
    print(f"Skipped:    {len(skipped)}")
    print()
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()