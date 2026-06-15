"""
Precompute ESM-2 variant-effect scores for TP53 (one-time, torch-enabled host).

Runs the ESM-2 protein language model over every possible single-residue
substitution in TP53 using *masked-marginal* scoring, and writes a small JSON
matrix that the app then serves OFFLINE with no torch at runtime:

    data/esm2_tp53_effect.json

Run this on a machine that has a GPU or patience (a free Colab works well):

    pip install torch transformers
    python tools/precompute_esm2.py                 # default 150M model
    python tools/precompute_esm2.py --model facebook/esm2_t33_650M_UR50D
    python tools/precompute_esm2.py --positions 175 248 273   # quick subset test

Scoring (per position i, substitution wt->aa), masked-marginal:
    score = logP(aa | seq with position i masked) - logP(wt | same)
More negative = the model finds the mutant less likely = more deleterious.

The reference sequence is fetched live from UniProt (P04637) so it is
authoritative and never hand-transcribed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

UNIPROT_TP53 = "P04637"
AA = "ACDEFGHIKLMNPQRSTVWY"          # 20 standard amino acids
OUT_DEFAULT = Path("data/esm2_tp53_effect.json")


def fetch_tp53_sequence() -> str:
    """Fetch the canonical TP53 protein sequence from UniProt (P04637)."""
    import urllib.request
    url = f"https://rest.uniprot.org/uniprotkb/{UNIPROT_TP53}.fasta"
    with urllib.request.urlopen(url, timeout=30) as r:
        fasta = r.read().decode()
    seq = "".join(ln.strip() for ln in fasta.splitlines() if ln and not ln.startswith(">"))
    if not seq:
        raise RuntimeError("Empty sequence fetched from UniProt")
    return seq


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Precompute ESM-2 TP53 variant-effect matrix")
    p.add_argument("--model", default="facebook/esm2_t30_150M_UR50D",
                   help="HuggingFace ESM-2 checkpoint (150M default; 650M is stronger/slower)")
    p.add_argument("--out", default=str(OUT_DEFAULT), help="Output JSON path")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto")
    p.add_argument("--positions", nargs="*", type=int, default=None,
                   help="Only score these 1-based positions (quick test). Default: all.")
    args = p.parse_args(argv)

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError:
        print("ERROR: needs torch + transformers. Run: pip install torch transformers",
              file=sys.stderr)
        return 1

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else ("cpu" if args.device == "auto" else args.device))
    print(f"[precompute] model={args.model} device={device}")

    seq = fetch_tp53_sequence()
    print(f"[precompute] TP53 sequence length: {len(seq)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).eval().to(device)
    mask_id = tok.mask_token_id
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in AA}

    enc = tok(seq, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)   # [1, L+special]; CLS at index 0

    positions = args.positions or list(range(1, len(seq) + 1))
    scores = {}
    for n, i in enumerate(positions, 1):
        if not (1 <= i <= len(seq)):
            continue
        wt = seq[i - 1]
        masked = input_ids.clone()
        masked[0, i] = mask_id                  # token index i == protein position i
        with torch.no_grad():
            logits = model(input_ids=masked).logits
        logprobs = F.log_softmax(logits[0, i], dim=-1)
        wt_lp = logprobs[aa_ids[wt]].item()
        scores[str(i)] = {
            a: round(logprobs[aa_ids[a]].item() - wt_lp, 3)
            for a in AA if a != wt
        }
        if n % 25 == 0 or n == len(positions):
            print(f"[precompute] {n}/{len(positions)} positions scored")

    out = {
        "model": args.model,
        "uniprot": UNIPROT_TP53,
        "sequence": seq,
        "sequence_length": len(seq),
        "method": "masked_marginal_llr",
        "scores": scores,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"[precompute] wrote {out_path} ({len(scores)} positions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
