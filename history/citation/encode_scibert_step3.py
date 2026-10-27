"""Step 3 only: Encode concept names with SciBERT (names already fetched)."""
import json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DATA_DIR = '/root/autodl-tmp/openalex_data/pencil_data'

if __name__ == '__main__':
    t0 = time.time()

    concept_list = json.load(open(f'{DATA_DIR}/concept_ids.json'))
    concept_names = json.load(open(f'{DATA_DIR}/concept_names.json'))
    print(f"Concepts: {len(concept_list)}, names: {len(concept_names)}", flush=True)

    MODEL_DIR = f'{DATA_DIR}/scibert_model'
    print("Loading SciBERT from local...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.eval()

    embeddings = []
    batch_size = 256
    texts = [concept_names.get(c, 'unknown') for c in concept_list]

    print(f"Encoding {len(texts)} concepts...", flush=True)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=32,
                           return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_emb)
        if i % 5000 == 0:
            print(f"  encoded {i}/{len(texts)}", flush=True)

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embeddings shape: {embeddings.shape}", flush=True)

    torch.save(torch.from_numpy(embeddings).float(), f'{DATA_DIR}/concept_scibert.pt')
    sz = os.path.getsize(f'{DATA_DIR}/concept_scibert.pt') / 1e6
    print(f"Saved concept_scibert.pt ({sz:.1f}MB)", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)
    print("Done.", flush=True)
