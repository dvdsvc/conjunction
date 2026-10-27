"""Fast concept name fetching via Wikidata SPARQL."""
import json, requests, time

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'

cids = json.load(open(f'{DATA_DIR}/concept_ids.json'))
print(f"Total concepts: {len(cids)}", flush=True)

qids = ['Q' + c[1:] for c in cids]

names = {}
BATCH = 500
for i in range(0, len(qids), BATCH):
    batch = qids[i:i+BATCH]
    values = ' '.join([f'wd:{q}' for q in batch])
    query = f"""SELECT ?id ?label WHERE {{
      VALUES ?id {{ {values} }}
      ?id rdfs:label ?label .
      FILTER(LANG(?label) = "en")
    }}"""
    try:
        r = requests.post('https://query.wikidata.org/sparql',
                         data={'query': query},
                         headers={'Accept': 'application/json'},
                         timeout=60)
        if r.status_code == 200:
            for b in r.json()['results']['bindings']:
                qid = b['id']['value'].split('/')[-1]
                label = b['label']['value']
                cid = 'C' + qid[1:]
                names[cid] = label
            print(f"  batch {i}-{i+len(batch)}: {len(names)} names total", flush=True)
        else:
            print(f"  batch {i}: HTTP {r.status_code}", flush=True)
    except Exception as e:
        print(f"  batch {i}: error {e}", flush=True)
    time.sleep(0.5)

print(f"\nFetched {len(names)}/{len(cids)} names", flush=True)
with open(f'{DATA_DIR}/concept_names.json', 'w') as f:
    json.dump(names, f, indent=2)
print("Saved concept_names.json", flush=True)
