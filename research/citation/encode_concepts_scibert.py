"""
Fetch OpenAlex concept names and encode with SciBERT.
Saves: concept_embeddings.pt (32744, 768) + concept_names.json
"""

import json, os, time
import requests
import numpy as np
import torch
from multiprocessing import Pool
from transformers import AutoTokenizer, AutoModel

DATA_DIR = '/root/autodl-tmp/openalex_data'
OUT_DIR = '/root/autodl-tmp/openalex_data/pencil_data'
MIN_SCORE = 0.3
MAX_LEVEL = 3
N_WORKERS = 40


def extract_concept_ids():
    """Re-extract concept IDs from source papers (same logic as build_pencil_data.py)."""
    from collections import Counter

    all_concepts = set()
    for fname in ['climate_papers.jsonl', 'geology_papers.jsonl']:
        path = f'{DATA_DIR}/{fname}'
        print(f"  scanning {fname}...", flush=True)
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                for c in d.get('concepts', []):
                    cid = c['id']
                    if '/' in cid:
                        cid = cid.rsplit('/', 1)[-1]
                    if 1 <= c['level'] <= MAX_LEVEL and c['score'] >= MIN_SCORE:
                        all_concepts.add(cid)

    concept_list = sorted(all_concepts)
    print(f"  unique concepts: {len(concept_list)}", flush=True)
    return concept_list


def fetch_names_batch(concept_ids, batch_size=25):
    """Fetch concept names from OpenAlex API in batches."""
    names = {}
    for i in range(0, len(concept_ids), batch_size):
        batch = concept_ids[i:i + batch_size]
        filter_str = '|'.join(batch)
        url = f'https://api.openalex.org/concepts?filter=openalex:{filter_str}&per_page={batch_size}'
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                for c in resp.json().get('results', []):
                    oid = c['id'].rsplit('/', 1)[-1]
                    names[oid] = c.get('display_name', '')
        except:
            pass

        if i % 2500 == 0 and i > 0:
            print(f"    fetched {len(names)}/{i} names", flush=True)

    print(f"  fetched {len(names)}/{len(concept_ids)} names", flush=True)
    return names


def encode_scibert(concept_list, concept_names):
    """Encode concept names with SciBERT."""
    print("Loading SciBERT...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model.eval()

    embeddings = []
    batch_size = 256
    texts = [concept_names.get(c, 'unknown') for c in concept_list]

    print(f"  encoding {len(texts)} concepts...", flush=True)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=32,
                           return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_emb)

        if i % 5000 == 0:
            print(f"    encoded {i}/{len(texts)}", flush=True)

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"  embeddings shape: {embeddings.shape}", flush=True)
    return embeddings


if __name__ == '__main__':
    t0 = time.time()

    # Step 1: Extract concept IDs
    print("Step 1: Extracting concept IDs...", flush=True)
    concept_list = extract_concept_ids()

    # Save concept IDs
    with open(f'{OUT_DIR}/concept_ids.json', 'w') as f:
        json.dump(concept_list, f)
    print(f"  saved concept_ids.json ({len(concept_list)} concepts)", flush=True)

    # Step 2: Fetch names
    print("\nStep 2: Fetching concept names from OpenAlex...", flush=True)
    concept_names = fetch_names_batch(concept_list)

    # Save names
    with open(f'{OUT_DIR}/concept_names.json', 'w') as f:
        json.dump(concept_names, f, indent=2)
    print(f"  saved concept_names.json", flush=True)

    # Step 3: SciBERT encoding
    print("\nStep 3: Encoding with SciBERT...", flush=True)
    embeddings = encode_scibert(concept_list, concept_names)

    # Save embeddings
    torch.save(torch.from_numpy(embeddings).float(), f'{OUT_DIR}/concept_scibert.pt')
    sz = os.path.getsize(f'{OUT_DIR}/concept_scibert.pt') / 1e6
    print(f"  saved concept_scibert.pt ({sz:.1f}MB)", flush=True)

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
