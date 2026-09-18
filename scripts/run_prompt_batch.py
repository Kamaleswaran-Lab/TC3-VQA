# Replays a prompt from prompts/ over a JSONL manifest through the Anthropic Messages API. Each batch gets the
# filled prompt, its frames as images and its items as JSON; the JSON reply is written to <out_dir>/batch_<k>.json.
import argparse, base64, json, mimetypes, os, re, sys, time


def load_prompt(path):
    text = open(path).read()
    blocks = re.findall(r"```\n(.*?)\n```", text, re.S)
    if not blocks:
        sys.exit(f"no prompt block in {path}")
    return blocks


def image_block(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def frames_of(item, frames_dir):
    paths = item.get("frame_paths") or item.get("frames") or ([item["image_path"]] if item.get("image_path") else [])
    if not paths and item.get("file"):
        paths = [os.path.join(frames_dir, item["file"])]
    return [p if os.path.isabs(p) else os.path.join(frames_dir, p) for p in paths]


def fill(template, values):
    return re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), template)


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="prompts/<step>.md")
    ap.add_argument("manifest", help="JSONL, one item per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames-dir", default=".")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--block", type=int, default=0, help="which prompt block of the file to use (0 = first)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--extra", default="{}", help='JSON of extra placeholders, e.g. \'{"concepts": "..."}\'')
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    template = load_prompt(args.prompt)[args.block]
    items = [json.loads(l) for l in open(args.manifest) if l.strip()]
    extra = json.loads(args.extra)
    os.makedirs(args.out_dir, exist_ok=True)

    for k in range(0, len(items), args.batch_size):
        out = os.path.join(args.out_dir, f"batch_{k // args.batch_size:03d}.json")
        if os.path.exists(out):
            continue
        batch = items[k:k + args.batch_size]
        values = dict(extra, batch_file="(items below)", first=k, last=k + len(batch) - 1,
                      frames_dir=args.frames_dir, out_dir=args.out_dir, out_file=out)
        values.update({f: batch[0][f] for f in ("concept", "question") if len(batch) == 1 and f in batch[0]})
        content = []
        for it in batch:
            for p in frames_of(it, args.frames_dir):
                content.append(image_block(p))
        content.append({"type": "text", "text": fill(template, values) + "\n\nBatch items (JSON):\n" +
                        json.dumps(batch, ensure_ascii=False) + "\n\nReply with the JSON object only."})
        for attempt in range(3):
            try:
                msg = client.messages.create(model=args.model, max_tokens=args.max_tokens,
                                             messages=[{"role": "user", "content": content}])
                reply = "".join(b.text for b in msg.content if b.type == "text")
                parsed = parse_json(reply)
                if parsed is None:
                    raise ValueError("reply is not JSON")
                json.dump(parsed, open(out, "w"), indent=1, ensure_ascii=False)
                break
            except Exception as e:                      # rate limits and malformed replies: retry, then report
                if attempt == 2:
                    print(f"batch {k}: failed ({e})", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        print(f"batch {k // args.batch_size}: {len(batch)} items -> {out}", flush=True)


if __name__ == "__main__":
    main()
