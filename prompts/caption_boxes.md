# Text-overlay boxes for caption masking

Step: Refinement: frames flagged for caption leakage; the boxes are applied by `tccc_vqa/annotation/apply_caption_masks.py`.

Input: One item JSON: `item_id`, `frame_paths`.

Output: JSON with `item_id` and `frame_boxes`, one list of normalized `[x0, y0, x1, y1]` boxes per frame.

The prompt follows as it was run; the batch instructions inside it are handled by scripts/run_prompt_batch.py.

```
Locate burned-in TEXT overlays in video frames so they can be masked out.

STEP 1: Read the JSON at {batch_file} — it has item_id and frame_paths (a list of image files).
STEP 2: Read EVERY image in frame_paths (use Read on each path), in order.
STEP 3: For EACH frame, return a list of tight bounding boxes covering ALL clearly-readable overlaid/burned-in text: captions, subtitles, instructional text, on-screen labels/arrows-with-text, timers, watermarks, logos, channel bugs, and unit/name patches with legible text. Include anything a viewer could READ. Do NOT box blood, equipment, or anatomy that has no text.

Coordinates: NORMALIZED to [0,1], format [x0, y0, x1, y1] = top-left to bottom-right (x = horizontal fraction of width, y = vertical fraction of height). Make each box slightly generous so it fully covers the glyphs.

Return schema: item_id (copy from file) and frame_boxes = an array aligned to frame_paths order; entry k = list of boxes for frame k (empty list [] if that frame has no readable text). Be thorough — missing a text region means answer leakage survives.
```
