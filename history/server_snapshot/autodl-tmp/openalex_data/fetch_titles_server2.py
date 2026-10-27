#!/usr/bin/env python3
"""Fetch missing paper titles from OpenAlex API (server, requests + threads).
Polite pool: 10 req/s. Resumes from existing titles.jsonl.
"""
import json, time, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

EMAIL = "test@example.com"
PAGE_SIZE = 50
WORKERS = 10
BATCH_SAVE = 50_000

IDS_PATH = "/root/autodl-tmp/openalex_data/paper_ids.txt"
OUT_PATH = "/root/autodl-tmp/openalex_data/titles.jsonl"


def load_ids():
    with open(IDS_PATH) as f:
        return [line.strip() for line in f if line.strip()]


def fetch_batch(batch_ids, stats):
    filter_str = "openalex:" + "|".join(batch_ids)
    params = {
        "filter": filter_str,
        "per-page": PAGE_SIZE,
        "select": "id,title",
        "mailto": EMAIL,
    }
    for attempt in range(6):
        try:
            r = requests.get("https://api.openalex.org/works",
                             params=params, timeout=30)
            if r.status_code == 429:
                wait = max(float(r.headers.get("Retry-After", 5)), 5)
                stats["retries"] += 1
                time.sleep(wait)
                continue
            r.raise_for_status()
            results = []
            for w in r.json().get("results", []):
                oid = w["id"].split("/")[-1]
                t = w.get("title")
                if t:
                    results.append((oid, t))
            return results
        except Exception:
            if attempt == 5:
                stats["errors"] += 1
                return []
            time.sleep(2 ** attempt)
    return []


def main():
    # Load existing
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["id"])
    print(f"Already have {len(done):,} titles", flush=True)

    print("Loading IDs...", flush=True)
    all_ids = load_ids()
    ids = [pid for pid in all_ids if pid not in done]
    del all_ids
    print(f"  {len(ids):,} to fetch", flush=True)

    if not ids:
        print("Nothing to do.")
        return

    batches = [ids[i:i+PAGE_SIZE] for i in range(0, len(ids), PAGE_SIZE)]
    print(f"  {len(batches):,} batches  workers={WORKERS}", flush=True)

    stats = {"retries": 0, "errors": 0}
    total = len(ids)
    fetched = 0
    titles_buf = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_batch, b, stats): b for b in batches}

        for future in as_completed(futures):
            results = future.result()
            titles_buf.extend(results)
            fetched += PAGE_SIZE

            # Periodic save + progress
            if len(titles_buf) >= BATCH_SAVE:
                with open(OUT_PATH, "a") as f:
                    for oid, t in titles_buf:
                        f.write(json.dumps({"id": oid, "title": t}, ensure_ascii=False) + "\n")
                done.update(oid for oid, _ in titles_buf)
                titles_buf.clear()
                elapsed = time.time() - t0
                speed = fetched / max(elapsed, 1)
                eta = (total - fetched) / max(speed, 1) / 60
                print(f"  {fetched:,}/{total:,}  {speed:.0f}/s  "
                      f"ETA {eta:.1f}m  retries={stats['retries']}  "
                      f"total={len(done):,}", flush=True)

    # Final save
    if titles_buf:
        with open(OUT_PATH, "a") as f:
            for oid, t in titles_buf:
                f.write(json.dumps({"id": oid, "title": t}, ensure_ascii=False) + "\n")
        done.update(oid for oid, _ in titles_buf)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m  total={len(done):,}  "
          f"errors={stats['errors']}  retries={stats['retries']}", flush=True)


if __name__ == "__main__":
    main()
