"""Fetch concept names from OpenAlex API with high concurrency + optional proxy."""
import json, requests, time, sys, os
from multiprocessing import Pool

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
BATCH_SIZE = 25
N_WORKERS = 30
# Set to your Clash SOCKS5 proxy, e.g. 'socks5h://127.0.0.1:7891'
# or None to disable
PROXY = os.environ.get('FETCH_PROXY', None)

def fetch_batch(args):
    i, batch = args
    names = {}
    filter_str = '|'.join(batch)
    url = f'https://api.openalex.org/concepts?filter=openalex:{filter_str}&per_page={BATCH_SIZE}'
    proxies = {'https': PROXY, 'http': PROXY} if PROXY else None
    try:
        r = requests.get(url, timeout=30, proxies=proxies)
        if r.status_code == 200:
            for c in r.json().get('results', []):
                cid = c['id'].rsplit('/', 1)[-1]
                names[cid] = c.get('display_name', '')
    except Exception as e:
        pass
    return names

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    t0 = time.time()
    cids = json.load(open(f'{data_dir}/concept_ids.json'))
    print(f"Total concepts: {len(cids)}", flush=True)
    print(f"Workers: {N_WORKERS}, proxy: {PROXY}", flush=True)

    batches = [(i, cids[i:i+BATCH_SIZE]) for i in range(0, len(cids), BATCH_SIZE)]
    print(f"Batches: {len(batches)}", flush=True)

    all_names = {}
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(fetch_batch, batches)):
            all_names.update(result)
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(batches)} batches, {len(all_names)} names", flush=True)

    print(f"\nFetched {len(all_names)}/{len(cids)} names in {time.time()-t0:.1f}s", flush=True)
    with open(f'{data_dir}/concept_names.json', 'w') as f:
        json.dump(all_names, f)
    print("Saved concept_names.json", flush=True)
