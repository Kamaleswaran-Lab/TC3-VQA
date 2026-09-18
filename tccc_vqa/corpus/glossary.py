# Glossary chunks written by hand from the source documents. They spell out acronyms such as MARCH that the documents
# use without expanding, so that such queries can be retrieved.
from __future__ import annotations
from tccc_vqa.paths import CORPUS

import json
from pathlib import Path

from .chunk import Chunk

GLOSSARY_SOURCE_ID = "tccc_glossary_v1"

# Each entry becomes one retrieval chunk. Citation_id format mirrors the rest of the corpus.
# Sources for these definitions: TCCC Guidelines 25 Jan 2024, CoTCCC Skill Cards, ATP 4-02.11,
# JTS CPGs, NAEMT TCCC course materials — they all use these acronyms but rarely write the
# letter-by-letter expansion in a single passage.

_ENTRIES: list[tuple[str, str]] = [
    (
        "MARCH",
        "MARCH is the priority sequence for treating a combat casualty under TCCC:\n"
        "  M — Massive hemorrhage (life-threatening bleeding control: tourniquet, hemostatic dressing, wound packing, direct pressure).\n"
        "  A — Airway management (NPA, surgical airway as primary for far-forward providers when indicated).\n"
        "  R — Respiration / breathing (tension pneumothorax assessment, needle decompression, vented chest seal).\n"
        "  C — Circulation (IV/IO access, blood product / whole blood resuscitation, permissive hypotension).\n"
        "  H — Hypothermia prevention (active warming on every casualty regardless of ambient temperature).\n"
        "The order is the order of execution; perform earlier letters first, but reassess continuously.",
    ),
    (
        "MARCH-PAWS",
        "MARCH-PAWS extends MARCH with four additional steps that follow the life-threat priorities:\n"
        "  M, A, R, C, H — as in MARCH.\n"
        "  P — Pain management (Triple-Option Analgesia: OTFC lozenge for mild-moderate pain, ketamine for severe / shock, IV opioids when IV is in place).\n"
        "  A — Antibiotics (Moxifloxacin 400 mg PO if able to swallow; Ertapenem 1 g IV/IM otherwise, from the Combat Wound Medication Pack).\n"
        "  W — Wounds (cleaning, dressing of non-life-threatening wounds, burn coverage).\n"
        "  S — Splinting (immobilize fractures, eye shields, cervical immobilization where indicated).",
    ),
    (
        "SMARCH-PAWS",
        "SMARCH-PAWS adds Security as the first priority before MARCH-PAWS:\n"
        "  S — Security (return fire, take cover, move casualty to safer position before initiating care under fire).\n"
        "  M, A, R, C, H, P, A, W, S — as in MARCH-PAWS.\n"
        "SMARCH-PAWS reflects the combatant + medic dual role unique to TCCC.",
    ),
    (
        "MIST",
        "MIST is the TCCC handoff / report mnemonic, used during evacuation and casualty handover:\n"
        "  M — Mechanism of injury.\n"
        "  I — Injuries sustained (anatomic locations, severity).\n"
        "  S — Signs (vital signs, level of consciousness, perfusion).\n"
        "  T — Treatment given (interventions, medications, response to treatment).",
    ),
    (
        "TCCC phases",
        "TCCC defines three phases of care:\n"
        "  Care Under Fire (CUF): Care given while still under effective hostile fire. Priority: return fire, take cover, "
        "stop life-threatening external hemorrhage with a tourniquet high-and-tight. Minimal treatment beyond hemorrhage control.\n"
        "  Tactical Field Care (TFC): Care given when no longer under effective hostile fire but still in tactical environment. "
        "Full MARCH-PAWS reassessment and treatment.\n"
        "  Tactical Evacuation Care (TACEVAC): Care during evacuation to higher level of care. Includes en-route monitoring, "
        "continued interventions, additional resources available from the evacuation platform.",
    ),
    (
        "Triple-Option Analgesia",
        "TCCC's Triple-Option Analgesia for pain management:\n"
        "  Option 1 — Mild to moderate pain, conscious casualty able to fight: Acetaminophen 1000 mg PO + Meloxicam 15 mg PO (combat pill pack).\n"
        "  Option 2 — Moderate to severe pain, NOT in shock, NOT in respiratory distress: OTFC (Oral Transmucosal Fentanyl Citrate) "
        "800 mcg lozenge placed buccally and secured to the collar with a safety pin.\n"
        "  Option 3 — Severe pain in shock or respiratory distress: Ketamine 50 mg IM or 20 mg IV/IO, repeated as needed.",
    ),
    (
        "9-Line MEDEVAC",
        "9-Line MEDEVAC is the standard NATO/US military request for medical evacuation. Lines:\n"
        "  Line 1 — Location of pickup site (grid coordinates).\n"
        "  Line 2 — Radio frequency, call sign, suffix.\n"
        "  Line 3 — Number of patients by precedence (A urgent, B urgent-surgical, C priority, D routine, E convenience).\n"
        "  Line 4 — Special equipment required (A none, B hoist, C extraction, D ventilator).\n"
        "  Line 5 — Number of patients by type (L litter, A ambulatory).\n"
        "  Line 6 — Security of pickup site (peacetime: N none, P pediatric, etc.; wartime: N no enemy, P possible, E enemy, X armed escort).\n"
        "  Line 7 — Method of marking pickup site.\n"
        "  Line 8 — Patient nationality and status.\n"
        "  Line 9 — NBC contamination (wartime) or terrain description (peacetime).",
    ),
    (
        "Triad of Death",
        "The lethal triad of trauma — three mutually reinforcing physiological derangements that drive trauma mortality:\n"
        "  Hypothermia — impairs platelet function and coagulation factor activity.\n"
        "  Acidosis — depresses cardiac function and worsens coagulopathy.\n"
        "  Coagulopathy — perpetuates hemorrhage and prevents clot formation.\n"
        "TCCC's emphasis on aggressive hypothermia prevention and blood-product resuscitation is driven by this triad.",
    ),
    (
        "CAT tourniquet",
        "Combat Application Tourniquet (CAT) — the windlass-based limb tourniquet adopted as the TCCC-recommended "
        "limb tourniquet for combat casualty care. Generation 7 is the current standard. Apply 2-3 inches above the wound "
        "on bare skin; tighten until distal pulse is absent and bleeding stops; secure the windlass and time-mark with the "
        "application time. Tourniquet conversion (re-evaluation, possibly replacing with a pressure dressing) is considered "
        "after 2 hours if tactical situation permits and the casualty is not in shock.",
    ),
    (
        "Combat Wound Medication Pack (CWMP)",
        "The CWMP is the pre-packaged TCCC medication set carried by combatants and combat lifesavers:\n"
        "  Moxifloxacin 400 mg tablet (oral antibiotic, swallow if able).\n"
        "  Acetaminophen 1000 mg tablet (analgesic).\n"
        "  Meloxicam 15 mg tablet (NSAID analgesic).\n"
        "Given as soon as possible after wounding to casualties able to take oral medications.",
    ),
    (
        "Prolonged Casualty Care (PCC)",
        "Prolonged Casualty Care (PCC), formerly Prolonged Field Care (PFC), refers to TCCC delivered beyond the typical "
        "evacuation window (4-24 hours up to several days) when evacuation is delayed. Adds nursing-intensive interventions "
        "such as urinary catheter, NG/OG tube, mechanical ventilation if available, blood product re-dosing, infection "
        "monitoring, and continuous reassessment. Distinct emphasis on team-based care and casualty documentation.",
    ),
]


def _build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, (term, definition) in enumerate(_ENTRIES):
        cid = f"{GLOSSARY_SOURCE_ID}:{term.lower().replace(' ', '_').replace('-', '_')}:{idx:03d}"
        chunks.append(Chunk(
            citation_id=cid,
            source_id=GLOSSARY_SOURCE_ID,
            text=f"{term}\n\n{definition}",
            page_start=1,
            page_end=1,
            section_path=term,
            chunk_idx=idx,
            extra={"hand_authored": True, "version": "v1"},
        ))
    return chunks


def write_glossary_chunks(processed_dir: Path) -> Path:
    """Write hand-authored TCCC glossary chunks to <processed_dir>/<source_id>/glossary.chunks.json.
    The next corpus index rebuild will pick these up automatically."""
    from dataclasses import asdict

    out_dir = Path(processed_dir) / GLOSSARY_SOURCE_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "glossary.chunks.json"
    chunks = _build_chunks()
    out_path.write_text(json.dumps(
        [asdict(c) for c in chunks], indent=2, ensure_ascii=False,
    ))
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path,
                   default=Path(CORPUS + "/processed"))
    args = p.parse_args()
    out = write_glossary_chunks(args.processed_dir)
    print(f"wrote {sum(1 for _ in _ENTRIES)} glossary chunks → {out}")
