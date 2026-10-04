# Builds one retrieval query per scene from the concept, its visual triggers, the visible evidence and the observation,
# and saves the BGE-large embeddings. Passage selection follows in select_passages.py.
from tc3_vqa.paths import CORPUS, WORK
import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer

IDX = CORPUS + "/index"
P = WORK
QPREFIX = "Represent this sentence for searching relevant passages: "

def main():
    scenes_file = sys.argv[1] if len(sys.argv) > 1 else f"{P}/pilot_scenes.json"
    emb_out = sys.argv[2] if len(sys.argv) > 2 else f"{P}/query_emb.npy"
    inv = {c["concept_id"]: c for c in json.load(open(f"{P}/concept_inventory.json"))}
    scenes = json.load(open(scenes_file))["scenes"]
    qs = []
    for s in scenes:
        trig = "; ".join(inv[s["concept"]]["visual_triggers"][:2]) if s["concept"] in inv else ""
        q = (f"{s['concept_readable']}. {trig}. {s.get('visible_evidence','')}. "
             f"{s.get('visual_observation','')}. body region {s.get('anatomy','')}.")
        qs.append(QPREFIX + q)
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    qe = model.encode(qs, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(emb_out, qe)
    print(f"[retrA] embedded {len(qs)} scene queries -> {emb_out} {qe.shape}", flush=True)

if __name__ == "__main__":
    main()
