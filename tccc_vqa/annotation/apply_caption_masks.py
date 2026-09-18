# Masks burned-in captions that reveal answers. Boxes near the top or bottom edge become full-width bands; other text
# boxes are padded. Input is a JSONL of frame paths with normalized text boxes.
import json, os, cv2

def expand(b):
    """normalized [x0,y0,x1,y1] -> mask rect. Caption banners (near top/bottom) become full-width bands."""
    if len(b) < 4: return None
    x0, y0, x1, y1 = b[:4]
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    near_top, near_bottom = y0 < 0.15, y1 > 0.85
    if near_top or near_bottom:
        X0, X1 = 0.0, 1.0
        Y0 = 0.0 if near_top else max(0.0, y0 - 0.04)
        Y1 = 1.0 if near_bottom else min(1.0, y1 + 0.04)
    else:
        X0, X1 = max(0, x0 - 0.03), min(1, x1 + 0.03)
        Y0, Y1 = max(0, y0 - 0.03), min(1, y1 + 0.03)
    return X0, Y0, X1, Y1

def mask_frame(src_path, boxes, out_path):
    img = cv2.imread(src_path)
    if img is None: return False
    H, W = img.shape[:2]
    for b in boxes:
        r = expand(b)
        if not r: continue
        X0, Y0, X1, Y1 = r
        cv2.rectangle(img, (int(X0 * W), int(Y0 * H)), (int(X1 * W), int(Y1 * H)), (0, 0, 0), -1)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True

# Order of operations: the visual audit marks frames whose burned-in captions reveal an answer; the text-box
# prompt (prompts/caption_boxes.md) returns the boxes for those frames; this script masks them into masked_frames/,
# and the items keep the masked frame paths.

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("boxes", help="JSONL: {\"frame_path\": ..., \"boxes\": [[x0,y0,x1,y1], ...]}")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for line in open(a.boxes):
        r = json.loads(line)
        out = os.path.join(a.out_dir, os.path.basename(r["frame_path"]))
        n += mask_frame(r["frame_path"], r.get("boxes") or [], out)
    print(f"masked {n} frames -> {a.out_dir}")


if __name__ == "__main__":
    main()
