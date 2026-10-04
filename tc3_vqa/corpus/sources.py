# Registry of the corpus sources: identifier, name, URL, version and licence policy.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FetchMethod = Literal["http", "manual", "api"]
Policy = Literal["full_text", "cite_only"]


@dataclass
class Source:
    id: str                           # snake_case unique id (used as citation_id prefix)
    name: str                         # human-readable name
    tier: int                         # 1 = Tier 1, ...
    fetch: FetchMethod                # http = direct download, api = NCBI/arXiv, manual = put files in by hand
    policy: Policy = "full_text"      # cite_only stores only metadata, allows only citation usage
    urls: list[str] = field(default_factory=list)
    version: str | None = None        # date/version for sources that update (e.g. TCCC Guidelines)
    notes: str = ""


# ── Tier 1 (full text, auto-download targets) ─────────────────────────────────
# URLs were verified manually; some need re-verification when the publisher rotates paths.
TIER1: list[Source] = [
    Source(
        id="tccc_guidelines_2024_01_25",
        name="TCCC Guidelines (25 Jan 2024)",
        tier=1,
        fetch="http",
        version="2024-01-25",
        notes="J Spec Oper Med 24(1):100-108. allogy CDN direct PDF, verified (291 KB).",
        urls=[
            "https://learning-media.allogy.com/api/v1/pdf/f4cf1d4e-3191-443a-befc-415838fb04f2/contents",
        ],
    ),
    Source(
        id="tccc_guidelines_medical_personnel",
        name="TCCC Guidelines for Medical Personnel",
        tier=1,
        fetch="http",
        notes="CoTCCC guidelines for medical personnel (TCCC-MP). allogy CDN, verified (295 KB).",
        urls=[
            "https://learning-media.allogy.com/api/v1/pdf/1045f287-baa4-4990-8951-de517a262ee2/contents",
        ],
    ),
    Source(
        id="jts_cpgs_a_h",
        name="JTS Clinical Practice Guidelines A-H (Complete Zip)",
        tier=1,
        fetch="http",
        notes="Official jts.health.mil distribution zip. Multiple PDFs + CPG_Index.pdf, verified (61 MB).",
        urls=[
            "https://jts.health.mil/assets/docs/cpgs/Zip_of_Current_JTS_CPGs_A-H.zip",
        ],
    ),
    Source(
        id="jts_cpgs_i_z",
        name="JTS Clinical Practice Guidelines I-Z (Complete Zip)",
        tier=1,
        fetch="http",
        notes="Sister zip to A-H. Verified (90 MB).",
        urls=[
            "https://jts.health.mil/assets/docs/cpgs/Zip_of_Current_JTS_CPGs_I-Z.zip",
        ],
    ),
    Source(
        id="jts_cpg_index",
        name="JTS CPG Index (master list)",
        tier=1,
        fetch="http",
        notes="Canonical ID/title/date mapping for every CPG. Used to standardize citation_id.",
        urls=[
            "https://jts.health.mil/assets/docs/cpgs/CPG_Index.pdf",
        ],
    ),
    Source(
        id="jts_top10_operational",
        name="JTS Top 10 Operational Readiness CPGs",
        tier=1,
        fetch="http",
        notes="Operationally prioritized CPG zip. Overlaps A-H/I-Z but useful for priority stratification.",
        urls=[
            "https://jts.health.mil/assets/docs/cpgs/Top_10_Operational_Readiness_CPGs.zip",
        ],
    ),
    Source(
        id="atp_4_02_11_casualty_response",
        name="ATP 4-02.11 — Casualty Response, Tactical Combat Casualty Care, and First Aid (US Army, 23 Mar 2026)",
        tier=1,
        fetch="http",
        version="2026-03-23",
        notes="MEDCoE 2026 consolidation that supersedes TC 4-02.1 and FM 4-25.11. "
              "Announcement: https://www.army.mil/article/291504. PDF downloadable "
              "after our crawler started sending User-Agent + Referer headers (armypubs requires them).",
        urls=[
            "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN46159-ATP_4-02.11-000-WEB-1.pdf",
        ],
    ),
    Source(
        id="fm_4_25_11_army_first_aid",
        name="FM 4-25.11 — First Aid (US Army)",
        tier=1,
        fetch="manual",
        notes="armypubs.army.mil requires browser cookies, so direct download fails. "
              "Browser-download from https://armypubs.army.mil/ProductMaps/PubForm/Details.aspx?PUB_ID=80516 "
              "into corpus_raw/fm_4_25_11_army_first_aid/. Superseded by ATP 4-02.11 (consolidated); kept here "
              "for completeness only.",
        urls=[],
    ),
    Source(
        id="tc_4_02_1_first_aid",
        name="TC 4-02.1 — First Aid (US Army, includes FM 4-25.11/NTRP 4-02.1.1)",
        tier=1,
        fetch="manual",
        notes="armypubs.army.mil. Candidate PDF URL: "
              "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN14135-TC_4-02.1-002-WEB-3.pdf "
              "Browser-download into corpus_raw/tc_4_02_1_first_aid/. "
              "Superseded by ATP 4-02.11 (consolidated).",
        urls=[],
    ),
    Source(
        id="jp_4_02_joint_health_services",
        name="Joint Publication 4-02 — Joint Health Services",
        tier=1,
        fetch="manual",
        notes="jcs.mil portal returns 403. Candidate URL: "
              "https://www.jcs.mil/Portals/36/Documents/Doctrine/pubs/jp4_02ch1.pdf "
              "Browser-download into corpus_raw/jp_4_02_joint_health_services/.",
        urls=[],
    ),
    Source(
        id="cotccc_skill_cards",
        name="CoTCCC Skill Cards / MARCH cards",
        tier=1,
        fetch="manual",
        notes="deployedmedicine.com requires login. Download with your account into corpus_raw/.",
        urls=[],
    ),
    Source(
        id="dha_clinical_practice_guidance",
        name="DHA Clinical Practice Guidance (TCCC-related subset)",
        tier=1,
        fetch="manual",
        notes="Filter health.mil resources down to TCCC-relevant items, then drop into corpus_raw/.",
        urls=[],
    ),
]

# ── Tier 2 (cite-only) ────────────────────────────────────────────────────────
TIER2: list[Source] = [
    Source(id="naemt_tccc_textbook",   name="NAEMT TCCC Textbook (commercial)",
           tier=2, fetch="manual", policy="cite_only"),
    Source(id="special_ops_medical_handbook", name="Special Operations Medical Handbook",
           tier=2, fetch="manual", policy="cite_only"),
    Source(id="ranger_medic_handbook", name="Ranger Medic Handbook",
           tier=2, fetch="manual", policy="cite_only",
           notes="If a DoD public-release version exists, can be promoted to Tier 1."),
    Source(id="sofmh", name="Special Operations Forces Medical Handbook",
           tier=2, fetch="manual", policy="cite_only"),
    Source(id="phtls", name="PHTLS (Prehospital Trauma Life Support)",
           tier=2, fetch="manual", policy="cite_only"),
    Source(id="atls", name="Advanced Trauma Life Support (ACS)",
           tier=2, fetch="manual", policy="cite_only"),
]

# ── Tier 3 (academic literature, API crawl in its own modules) ────────────────
TIER3: list[Source] = [
    Source(id="pubmed", name="PubMed (TCCC-related)", tier=3, fetch="api",
           notes="NCBI E-utilities. Query config lives in corpus_sources.md."),
    Source(id="arxiv", name="arXiv (TCCC + combat medical AI)", tier=3, fetch="api"),
]

ALL_SOURCES: list[Source] = TIER1 + TIER2 + TIER3


def get_source(source_id: str) -> Source:
    for s in ALL_SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(source_id)


def downloadable_http_sources() -> list[Source]:
    """Return http-fetch sources that have a non-empty URL list."""
    return [s for s in ALL_SOURCES if s.fetch == "http" and s.urls]
