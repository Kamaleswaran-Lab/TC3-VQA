# Computes frozen image-encoder features for every released frame so the benchmark can be used where the source
# videos are no longer reachable. Writes frame_features.npz (frame ids and one float16 matrix per encoder) and a
# JSON describing the encoders into the artifacts directory; build_deposit.py copies both into the package.
from tccc_vqa.paths import FRAMES, RELEASE, RELEASE_PARENT
import argparse, json, os
import numpy as np
import torch
from PIL import Image

DEP = RELEASE
ART = RELEASE_PARENT + '/artifacts'
ENCODERS = {'clip_vit_l14_336': 'openai/clip-vit-large-patch14-336',
            'siglip_so400m_384': 'google/siglip-so400m-patch14-384',
            'dinov2_large': 'facebook/dinov2-large'}


def frame_ids():
    ids = []
    for line in open(f'{DEP}/data/items.jsonl'):
        for f in json.loads(line)['frame_refs']:
            if f['frame_id'] not in ids:
                ids.append(f['frame_id'])
    return ids


def embed(model_id, ids, batch, device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    vision = getattr(model, 'vision_model', model)
    out = []
    for i in range(0, len(ids), batch):
        imgs = [Image.open(f'{FRAMES}/{f}').convert('RGB') for f in ids[i:i + batch]]
        px = proc(images=imgs, return_tensors='pt')['pixel_values'].to(device, torch.float16)
        with torch.no_grad():
            h = vision(pixel_values=px)
        v = h.pooler_output if getattr(h, 'pooler_output', None) is not None else h.last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy().astype(np.float16))
        print(f'  {model_id}: {min(i + batch, len(ids))}/{len(ids)}', flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoders', default=','.join(ENCODERS), help='comma-separated subset of the encoder keys')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ids = frame_ids()
    arrays = {'frame_ids': np.array(ids)}
    meta = {'frames': len(ids), 'note': 'L2-normalized pooled image features, float16, rows aligned with frame_ids',
            'encoders': {}}
    for key in args.encoders.split(','):
        model_id = ENCODERS[key]
        arrays[key] = embed(model_id, ids, args.batch, device)
        meta['encoders'][key] = {'model': model_id, 'dim': int(arrays[key].shape[1])}
        print(f'{key}: {arrays[key].shape}', flush=True)
    np.savez_compressed(f'{ART}/frame_features.npz', **arrays)
    json.dump(meta, open(f'{ART}/frame_features.json', 'w'), indent=1)
    size = os.path.getsize(f'{ART}/frame_features.npz') / 1e6
    print(f'wrote {ART}/frame_features.npz ({size:.1f} MB) for {len(ids)} frames')


if __name__ == '__main__':
    main()
