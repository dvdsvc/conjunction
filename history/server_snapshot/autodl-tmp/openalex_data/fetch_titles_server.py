#!/usr/bin/env python3
"""Fetch missing paper titles from OpenAlex API (server, no proxy).
Polite pool: 10 req/s with email. Resumes from existing titles.jsonl.
"""
import asyncio, aiohttp, json, time

EMAIL = "test@example.com"
PAGE_SIZE = 50
WORKERS = 10
BATCH_DELAY = 0.15
BATCH_SAVE = 50_000

IDS_PATH = "/root/autodl-tmp/openalex_data/paper_ids.txt"
OUT_PATH = "/root/autodl-tmp/openalex_data/titles.jsonl"


def load_ids():
    with open(IDS_PATH) as f:
        return [line.strip() for line in f if line.strip()]


async def fetch_batch(session, batch_ids, titles, stats):
    filter_str = "openalex:" + "|".join(batch_ids)
    params = {
        "filter": filter_str,
        "per-page": PAGE_SIZE,
        "select": "id,title",
        "mailto": EMAIL,
    }
    for attempt in range(6):
        try:
            async with session.get("https://api.openalex.org/works",
                                   params=params,
                                   timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    wait = max(float(resp.headers.get("Retry-After", 5)), 5)
                    stats["retries"] += 1
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                for w in data.get("results", []):
                    oid = w["id"].split("/")[-1]
                    t = w.get("title")
                    if t:
                        titles[oid] = t
                return
        except Exception:
            if attempt == 5:
                stats["failed_ids"].extend(batch_ids)
                stats["errors"] += 1
            else:
                await asyncio.sleep(2 ** attempt)


async def worker(queue, session, titles, stats):
    while True:
        batch = await queue.get()
        if batch is None:
            queue.task_done()
            break
        await fetch_batch(session, batch, titles, stats)
        stats["fetched"] += len(batch)
        queue.task_done()


def save_titles(titles, path):
    with open(path, "a") as f:
        for oid, t in titles.items():
            f.write(json.dumps({"id": oid, "title": t}, ensure_ascii=False) + "\n")


async def main():
    done = set()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass
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
    print(f"  {len(batches):,} batches  workers={WORKERS}  (polite pool, no proxy)", flush=True)

    titles = {}
    stats = {"fetched": 0, "retries": 0, "failed_ids": [], "errors": 0}
    total = len(ids)
    t0 = time.time()

    connector = aiohttp.TCPConnector(limit=WORKERS, limit_per_host=WORKERS)
    async with aiohttp.ClientSession(connector=connector) as session:
        queue = asyncio.Queue(maxsize=WORKERS * 2)
        workers = [asyncio.create_task(worker(queue, session, titles, stats))
                   for _ in range(WORKERS)]

        for i in range(len(batches)):
            await queue.put(batches[i])

            if i % WORKERS == 0 and i > 0:
                await asyncio.sleep(BATCH_DELAY)

            if len(titles) >= BATCH_SAVE:
                save_titles(titles, OUT_PATH)
                done.update(titles.keys())
                titles.clear()
                n = stats["fetched"]
                elapsed = time.time() - t0
                speed = n / max(elapsed, 1)
                eta = (total - n) / max(speed, 1) / 60
                print(f"  {n:,}/{total:,}  {speed:.0f}/s  "
                      f"ETA {eta:.1f}m  retries={stats['retries']}  "
                      f"total={len(done):,}", flush=True)

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    if titles:
        save_titles(titles, OUT_PATH)
        done.update(titles.keys())

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m  total={len(done):,}  "
          f"errors={stats['errors']}  failed_ids={len(stats['failed_ids']):,}")


if __name__ == "__main__":
    asyncio.run(main())
