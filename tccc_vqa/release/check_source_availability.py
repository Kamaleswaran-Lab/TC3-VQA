# Checks whether the source videos of the release are still reachable, so the paper can report how much of the
# footage a reader can retrieve today. Queries each video with yt-dlp metadata only and writes a per-video status
# CSV plus a summary.
from tccc_vqa.paths import EXPERIMENTS, RELEASE
import argparse, csv, json, subprocess, sys
from collections import Counter

STATUS_HINTS = [('Private video', 'private'), ('This video is unavailable', 'unavailable'),
                ('has been removed', 'removed'), ('terminated', 'channel_terminated'),
                ('Sign in to confirm your age', 'age_restricted'), ('confirm you are not a bot', 'login_required'),
                ('Video unavailable', 'unavailable')]


def probe(video_id, cookies, timeout):
    cmd = ['yt-dlp', '--skip-download', '--print', '%(duration)s|%(availability)s', '--no-warnings']
    if cookies:
        cmd += ['--cookies', cookies]
    cmd.append(f'https://www.youtube.com/watch?v={video_id}')
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 'timeout', ''
    if p.returncode == 0 and p.stdout.strip():
        duration, availability = (p.stdout.strip().split('|') + [''])[:2]
        return 'available', f'{duration}s {availability}'.strip()
    err = (p.stderr or '').strip().replace('\n', ' ')
    for hint, status in STATUS_HINTS:
        if hint.lower() in err.lower():
            return status, err[-200:]
    return 'error', err[-200:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=RELEASE + '/meta/source_videos.csv')
    ap.add_argument('--out', default=EXPERIMENTS + '/source_availability.csv')
    ap.add_argument('--cookies', default=None, help='cookies.txt, needed for the age-restricted sources')
    ap.add_argument('--timeout', type=int, default=60)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.manifest)))
    out = []
    for i, r in enumerate(rows, 1):
        status, detail = probe(r['video_id'], args.cookies, args.timeout)
        out.append({'video_id': r['video_id'], 'license_category': r['license_category'],
                    'n_items': r['n_items'], 'status': status, 'detail': detail})
        print(f"[{i}/{len(rows)}] {r['video_id']} {status}", flush=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['video_id', 'license_category', 'n_items', 'status', 'detail'])
        w.writeheader(); w.writerows(out)
    by_status = Counter(o['status'] for o in out)
    items = Counter()
    for o in out:
        items[o['status']] += int(o['n_items'] or 0)
    print(json.dumps({'videos': by_status, 'items': items}, indent=1))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
