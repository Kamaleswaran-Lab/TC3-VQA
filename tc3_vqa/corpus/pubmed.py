# Retrieves PubMed records for the auxiliary literature set through the NCBI E-utilities and stores title, abstract and
# metadata as chunks.
from __future__ import annotations
from tc3_vqa.paths import CORPUS

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import requests

from .chunk import Chunk

PUBMED_SOURCE_ID = "pubmed_tccc_v1"
PUBMED_CIVILIAN_SOURCE_ID = "pubmed_civilian_trauma_v1"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_UA = {"User-Agent": "TC3-VQA-corpus-builder/1.0 (research)"}

# Stratified TCCC queries — kept narrow per query so the union of results spans the operationally relevant
# subspaces (massive hemorrhage, airway, respiration, circulation, hypothermia, evacuation, doctrine).
TCCC_QUERIES: list[tuple[str, str]] = [
    ("tccc_general",
     '("tactical combat casualty care"[Title/Abstract] OR "TCCC"[Title/Abstract] OR '
     '"battlefield trauma"[Title/Abstract] OR "combat casualty"[Title/Abstract] OR '
     '"point of injury"[Title/Abstract] OR "prehospital combat"[Title/Abstract])'),
    ("tourniquet",
     '("tourniquet"[Title/Abstract] OR "CAT tourniquet"[Title/Abstract] OR '
     '"junctional tourniquet"[Title/Abstract]) AND '
     '("combat"[Title/Abstract] OR "military"[Title/Abstract] OR "battlefield"[Title/Abstract])'),
    ("needle_decompression",
     '("needle decompression"[Title/Abstract] OR "needle thoracostomy"[Title/Abstract] OR '
     '"tension pneumothorax"[Title/Abstract]) AND '
     '("prehospital"[Title/Abstract] OR "combat"[Title/Abstract] OR "military"[Title/Abstract])'),
    ("airway_combat",
     '("cricothyroidotomy"[Title/Abstract] OR "surgical airway"[Title/Abstract] OR '
     '"NPA"[Title/Abstract] OR "extraglottic airway"[Title/Abstract]) AND '
     '("combat"[Title/Abstract] OR "tactical"[Title/Abstract] OR "battlefield"[Title/Abstract])'),
    ("tranexamic_acid_combat",
     '"tranexamic acid"[Title/Abstract] AND '
     '("combat"[Title/Abstract] OR "battlefield"[Title/Abstract] OR "military trauma"[Title/Abstract])'),
    ("whole_blood_resuscitation",
     '("whole blood"[Title/Abstract] OR "low-titer"[Title/Abstract] OR "blood transfusion"[Title/Abstract]) '
     'AND ("combat"[Title/Abstract] OR "prehospital"[Title/Abstract] OR "battlefield"[Title/Abstract])'),
    ("hypothermia_combat",
     '("hypothermia"[Title/Abstract] OR "thermal management"[Title/Abstract] OR "rewarming"[Title/Abstract]) '
     'AND ("combat"[Title/Abstract] OR "trauma-induced"[Title/Abstract] OR "battlefield"[Title/Abstract])'),
    ("hemostatic_dressing",
     '("hemostatic dressing"[Title/Abstract] OR "Combat Gauze"[Title/Abstract] OR '
     '"Chitosan dressing"[Title/Abstract] OR "QuikClot"[Title/Abstract])'),
    ("prolonged_field_care",
     '("prolonged field care"[Title/Abstract] OR "prolonged casualty care"[Title/Abstract] OR '
     '"austere medicine"[Title/Abstract]) AND '
     '("combat"[Title/Abstract] OR "military"[Title/Abstract] OR "operational"[Title/Abstract])'),
    ("medevac_evacuation",
     '("MEDEVAC"[Title/Abstract] OR "tactical evacuation"[Title/Abstract] OR '
     '"CASEVAC"[Title/Abstract]) AND '
     '("combat"[Title/Abstract] OR "military"[Title/Abstract])'),
    ("pain_combat",
     '("ketamine"[Title/Abstract] OR "OTFC"[Title/Abstract] OR "fentanyl lozenge"[Title/Abstract] OR '
     '"battlefield analgesia"[Title/Abstract]) AND '
     '("combat"[Title/Abstract] OR "military"[Title/Abstract] OR "prehospital"[Title/Abstract])'),
    ("burn_combat",
     '("burn"[Title/Abstract] AND ("combat"[Title/Abstract] OR "battlefield"[Title/Abstract] OR '
     '"military"[Title/Abstract])) AND ("resuscitation"[Title/Abstract] OR "management"[Title/Abstract])'),
    ("tbi_combat",
     '("traumatic brain injury"[Title/Abstract] OR "TBI"[Title/Abstract] OR "concussion"[Title/Abstract]) '
     'AND ("combat"[Title/Abstract] OR "blast"[Title/Abstract] OR "battlefield"[Title/Abstract])'),
    ("cbrn",
     '("CBRN"[Title/Abstract] OR "chemical warfare"[Title/Abstract] OR "biological warfare"[Title/Abstract]) '
     'AND "casualty"[Title/Abstract]'),
    ("preventable_death_combat",
     '("preventable death"[Title/Abstract] OR "battlefield mortality"[Title/Abstract] OR '
     '"died of wounds"[Title/Abstract]) AND ("combat"[Title/Abstract] OR "military"[Title/Abstract])'),
]

CIVILIAN_QUERIES: list[tuple[str, str]] = [
    ("atls_general",
     '"advanced trauma life support"[Title/Abstract] AND ("civilian"[Title/Abstract] OR "hospital"[Title/Abstract])'),
    ("phtls_general",
     '"prehospital trauma life support"[Title/Abstract]'),
    ("civilian_tourniquet",
     '"tourniquet"[Title/Abstract] AND ("civilian"[Title/Abstract] OR "EMS"[Title/Abstract] OR '
     '"prehospital trauma"[Title/Abstract])'),
    ("civilian_txa",
     '"tranexamic acid"[Title/Abstract] AND ("CRASH"[Title/Abstract] OR "civilian trauma"[Title/Abstract])'),
    ("civilian_chest_seal",
     '("chest seal"[Title/Abstract] OR "occlusive dressing"[Title/Abstract]) AND '
     '"penetrating chest"[Title/Abstract]'),
    ("civilian_decompression",
     '"needle thoracostomy"[Title/Abstract] AND '
     '("emergency department"[Title/Abstract] OR "trauma center"[Title/Abstract])'),
    ("civilian_damage_control",
     '"damage control resuscitation"[Title/Abstract] AND '
     '("trauma center"[Title/Abstract] OR "hospital"[Title/Abstract])'),
    ("crash_2_protocol",
     '("CRASH-2"[Title/Abstract] OR "CRASH 2"[Title/Abstract])'),
]


def esearch_pmids(query: str, retmax: int = 200, mindate: int = 2005) -> list[str]:
    url = f"{EUTILS}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        "mindate": str(mindate),
        "maxdate": "3000",
        "datetype": "pdat",
    }
    r = requests.get(url, params=params, headers=_UA, timeout=30)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", []) or []


def efetch_abstracts(pmids: Iterable[str], batch_size: int = 200, delay: float = 0.4) -> list[dict]:
    pmids = list(pmids)
    out: list[dict] = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        url = f"{EUTILS}/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        r = requests.get(url, params=params, headers=_UA, timeout=60)
        r.raise_for_status()
        out.extend(_parse_pubmed_xml(r.text))
        time.sleep(delay)
    return out


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    # Lightweight XML parsing — stdlib ElementTree.  PubMed XML is well-formed.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    records: list[dict] = []
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        title_el = art.find(".//ArticleTitle")
        abs_els = art.findall(".//Abstract/AbstractText")
        journal_el = art.find(".//Journal/Title")
        year_el = art.find(".//PubDate/Year") or art.find(".//PubDate/MedlineDate")
        mesh_els = art.findall(".//MeshHeading/DescriptorName")
        authors = []
        for a in art.findall(".//AuthorList/Author"):
            ln = a.findtext("LastName") or ""
            initials = a.findtext("Initials") or ""
            if ln:
                authors.append(f"{ln} {initials}".strip())

        # ElementTree element is "falsy" when it has no children, so use `is None` check.
        if pmid_el is None or title_el is None or not abs_els:
            continue
        # Concat structured abstract sections with their labels
        abs_parts = []
        for ab in abs_els:
            label = ab.attrib.get("Label", "")
            text = "".join(ab.itertext()).strip()
            if not text:
                continue
            abs_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abs_parts).strip()
        if len(abstract) < 100:
            continue
        records.append({
            "pmid": pmid_el.text,
            "title": "".join(title_el.itertext()).strip(),
            "abstract": abstract,
            "journal": journal_el.text if journal_el is not None else "",
            "year": year_el.text if year_el is not None else "",
            "authors": authors,
            "mesh": [m.text for m in mesh_els if m.text],
        })
    return records


def crawl_pubmed(
    queries: list[tuple[str, str]],
    raw_dir: Path,
    processed_dir: Path,
    source_id: str,
    retmax_per_query: int = 300,
) -> int:
    """For each (label, query) run esearch+efetch, union by PMID, write one big chunks file."""
    raw_dir = Path(raw_dir) / source_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(processed_dir) / source_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pmids: set[str] = set()
    per_query: dict[str, list[str]] = {}
    for label, q in queries:
        pmids = esearch_pmids(q, retmax=retmax_per_query)
        per_query[label] = pmids
        all_pmids.update(pmids)
        time.sleep(0.4)
        print(f"  {label:30s} {len(pmids):5d} PMIDs")

    print(f"  union: {len(all_pmids)} unique PMIDs")
    records = efetch_abstracts(sorted(all_pmids))
    print(f"  fetched {len(records)} records with abstracts")

    # Save raw json
    (raw_dir / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    (raw_dir / "per_query_pmids.json").write_text(json.dumps(per_query, indent=2))

    # Build chunks — one chunk per paper (title + abstract + MeSH).
    chunks: list[Chunk] = []
    for idx, rec in enumerate(records):
        section = re.sub(r"\W+", "_", rec["journal"])[:60] or "unknown_journal"
        cid = f"{source_id}:{section}:pmid_{rec['pmid']}"
        text_parts = [
            f"Title: {rec['title']}",
            f"Authors: {', '.join(rec['authors'][:6])}{' et al.' if len(rec['authors']) > 6 else ''}",
            f"Journal: {rec['journal']} ({rec['year']})",
        ]
        if rec["mesh"]:
            text_parts.append(f"MeSH: {', '.join(rec['mesh'][:10])}")
        text_parts.append("")
        text_parts.append(rec["abstract"])
        chunks.append(Chunk(
            citation_id=cid, source_id=source_id,
            text="\n".join(text_parts),
            page_start=1, page_end=1,
            section_path=rec["journal"], chunk_idx=idx,
            extra={"pmid": rec["pmid"], "year": rec["year"]},
        ))

    out_path = out_dir / "pubmed.chunks.json"
    out_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False))
    return len(chunks)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["tccc", "civilian", "both"], default="both")
    p.add_argument("--retmax", type=int, default=300, help="esearch retmax per sub-query")
    p.add_argument("--tccc-raw", type=Path, default=Path(CORPUS + "/raw"))
    p.add_argument("--tccc-processed", type=Path,
                   default=Path(CORPUS + "/processed"))
    p.add_argument("--civilian-raw", type=Path,
                   default=Path(CORPUS + "/raw"))
    p.add_argument("--civilian-processed", type=Path,
                   default=Path(CORPUS + "/processed"))
    args = p.parse_args()

    if args.target in ("tccc", "both"):
        print("\n=== TCCC PubMed crawl ===")
        n = crawl_pubmed(TCCC_QUERIES, args.tccc_raw, args.tccc_processed,
                         PUBMED_SOURCE_ID, retmax_per_query=args.retmax)
        print(f"TCCC PubMed chunks: {n}")

    if args.target in ("civilian", "both"):
        print("\n=== Civilian PubMed crawl ===")
        n = crawl_pubmed(CIVILIAN_QUERIES, args.civilian_raw, args.civilian_processed,
                         PUBMED_CIVILIAN_SOURCE_ID, retmax_per_query=args.retmax)
        print(f"Civilian PubMed chunks: {n}")
