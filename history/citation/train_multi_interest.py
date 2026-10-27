"""Train Multi-Interest Network — fully GPU, no Python loops in training."""
import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F

from multi_interest_model import MultiInterestModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/autodl-tmp/openalex_data/processed")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--max-refs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = args.device

    print("Loading data...", flush=True)
    ref_lists = torch.load(os.path.join(args.data_dir, "ref_lists.pt"),
                           map_location="cpu", weights_only=False)
    bridges = torch.load(os.path.join(args.data_dir, "lbd_bridges_real.pt"),
                         map_location="cpu", weights_only=False)
    title_embeddings = torch.load(os.path.join(args.data_dir, "title_embeddings.pt"),
                                  map_location="cpu", weights_only=False)
    fd = torch.load(os.path.join(args.data_dir, "features.pt"),
                    map_location="cpu", weights_only=False)
    features = fd["features"]
    title_dim = title_embeddings.size(1)

    # Pre-compute padded refs on CPU then move to GPU
    n_total = len(ref_lists)
    MAX_REFS = args.max_refs
    print(f"Pre-computing padded refs ({n_total} × {MAX_REFS})...", flush=True)
    refs_padded = torch.zeros(n_total, MAX_REFS, dtype=torch.long)
    refs_len = torch.zeros(n_total, dtype=torch.long)
    for i, refs in enumerate(ref_lists):
        n = min(len(refs), MAX_REFS)
        if n > 0:
            refs_padded[i, :n] = torch.tensor(refs[:n], dtype=torch.long)
        refs_len[i] = n
    del ref_lists  # free memory

    # Split bridges by year
    src_all = bridges["source_idx"]
    tgt_all = bridges["target_idx"]
    year_thr = (2021 - 1970) / 56.0
    year = features[src_all, 0]
    train_mask = year <= year_thr
    val_mask = year > year_thr

    train_src = src_all[train_mask].to(device)
    train_tgt = tgt_all[train_mask].to(device)
    val_src = src_all[val_mask].to(device)
    val_tgt = tgt_all[val_mask].to(device)
    del bridges, features

    # Move data to GPU
    title_embeddings = title_embeddings.to(device)
    refs_padded = refs_padded.to(device)
    refs_len = refs_len.to(device)
    n_train = len(train_src)
    n_val = len(val_src)
    print(f"GPU data: title={title_embeddings.shape} refs={refs_padded.shape}", flush=True)
    print(f"Train: {n_train:,}  Val: {n_val:,}", flush=True)

    # Model
    model = MultiInterestModel(
        title_dim=title_dim, d_model=args.d_model, k_max=args.k_max
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pre-shuffle train indices
    train_order = torch.randperm(n_train, device=device)

    best_val_loss = float("inf")
    ckpt_dir = os.path.join(args.data_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    bs = args.batch_size
    n_train_batches = (n_train + bs - 1) // bs
    n_val_batches = (n_val + bs - 1) // bs

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_order = torch.randperm(n_train, device=device)
        total_loss = 0.0
        t0 = time.time()

        for i in range(n_train_batches):
            idx = train_order[i * bs : (i + 1) * bs]
            B = len(idx)
            src = train_src[idx]
            tgt = train_tgt[idx]

            # Gather ref indices for this batch: (B, MAX_REFS)
            batch_refs = refs_padded[src]         # (B, MAX_REFS)
            seq_len = refs_len[src]               # (B,)
            max_len = int(seq_len.max())

            # Gather title embeddings: (B, max_len, title_dim)
            user_titles = title_embeddings[batch_refs[:, :max_len]]
            cand_titles = title_embeddings[tgt]   # (B, title_dim)

            logits = model(user_titles, seq_len, cand_titles) / 0.07
            labels = torch.arange(B, device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / n_train_batches
        elapsed = time.time() - t0

        # ── Val ──
        model.eval()
        val_loss_sum = 0.0
        r1_sum = 0.0
        with torch.no_grad():
            for i in range(n_val_batches):
                src = val_src[i * bs : (i + 1) * bs]
                tgt = val_tgt[i * bs : (i + 1) * bs]
                B = len(src)

                batch_refs = refs_padded[src]
                seq_len = refs_len[src]
                max_len = int(seq_len.max())
                user_titles = title_embeddings[batch_refs[:, :max_len]]
                cand_titles = title_embeddings[tgt]

                logits = model(user_titles, seq_len, cand_titles) / 0.07
                labels = torch.arange(B, device=device)
                val_loss_sum += F.cross_entropy(logits, labels).item()
                r1_sum += (logits.argmax(1) == labels).float().sum().item()

        val_loss = val_loss_sum / n_val_batches
        r1 = r1_sum / n_val

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  R@1={r1:.3f}  {elapsed:.0f}s", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "d_model": args.d_model,
                "k_max": args.k_max,
            }, os.path.join(ckpt_dir, "multi_interest_best.pt"))
            print(f"  → saved", flush=True)

    print(f"\nDone. Best val_loss={best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
