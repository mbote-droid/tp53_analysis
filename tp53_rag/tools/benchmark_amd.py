"""
============================================================
AMD Hardware Benchmark Harness
tools/benchmark_amd.py
============================================================
Run ONCE on the AMD Developer Cloud (or any ROCm host) to produce real
inference/throughput numbers, then commit data/amd_benchmark.json so the app
can display them offline.

What it measures (whatever is available on the host):
  1. Device report — ROCm/HIP availability, GPU name, torch version.
  2. A GPU compute microbenchmark (large matmul) — proves work ran on the
     AMD GPU and reports sustained TFLOP/s.
  3. Optional LLM round-trip latency against the active inference backend.

Usage (on the AMD Cloud GPU host):
    python tools/benchmark_amd.py --matmul 8192 --iters 30
    python tools/benchmark_amd.py --llm "Summarise TP53 R175H"   # optional

Honest by construction: it records exactly what ran on the box. If torch/ROCm
is absent it says so rather than inventing numbers.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable when run as `python tools/benchmark_amd.py`
# (Python otherwise only puts tools/ on the path, so `import agents` fails).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_PATH = Path("data/amd_benchmark.json")


def device_report() -> dict:
    """Probe the host for ROCm/HIP + GPU details. Degrades gracefully."""
    rep = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda_available"] = bool(torch.cuda.is_available())
        # ROCm builds expose the HIP version here.
        rep["hip_version"] = getattr(torch.version, "hip", None)
        rep["is_rocm"] = rep["hip_version"] is not None
        if torch.cuda.is_available():
            rep["device_name"] = torch.cuda.get_device_name(0)
            rep["device_count"] = torch.cuda.device_count()
    except Exception as e:
        rep["torch"] = None
        rep["error"] = f"torch unavailable: {e}"
    return rep


def matmul_benchmark(size: int = 8192, iters: int = 30) -> dict:
    """Sustained large matmul — reports TFLOP/s actually achieved on device."""
    try:
        import torch
    except Exception as e:
        return {"ran": False, "reason": f"torch unavailable: {e}"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    # Warmup
    for _ in range(3):
        _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    # 2*N^3 FLOPs per matmul.
    flops = 2.0 * (size ** 3) * iters
    tflops = flops / elapsed / 1e12
    return {"ran": True, "device": device, "matrix": size, "iters": iters,
            "seconds": round(elapsed, 4), "tflops": round(tflops, 2),
            "dtype": str(dtype)}


def llm_latency(prompt: str) -> dict:
    """Round-trip latency against the active inference backend (optional)."""
    try:
        from agents.rag_chain import _build_backend
        backend = _build_backend()
        t0 = time.perf_counter()
        out = backend.generate("You are a concise assistant.", prompt,
                               max_tokens=128)
        dt = time.perf_counter() - t0
        return {"ran": True, "backend": backend.__class__.__name__,
                "seconds": round(dt, 3), "chars_out": len(out or "")}
    except Exception as e:
        return {"ran": False, "reason": str(e)}


def vllm_throughput(model: str, prompts: int = 16,
                    max_tokens: int = 128) -> dict:
    """Measure generation throughput (tokens/sec) with vLLM on ROCm.

    vLLM serves models on AMD Instinct via ROCm; this batches several prompts
    through the offline engine and reports sustained tokens/sec. Honest: if
    vLLM is not installed it reports unavailable rather than inventing a number.
    """
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        return {"ran": False, "reason": f"vLLM not installed: {e}"}
    try:
        llm = LLM(model=model)
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        batch = ["Summarise the clinical significance of TP53 R175H."] * prompts
        t0 = time.perf_counter()
        outs = llm.generate(batch, sp)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        return {"ran": True, "model": model, "prompts": prompts,
                "seconds": round(elapsed, 3),
                "tokens_per_s": round(total_tokens / elapsed, 1),
                "total_tokens": total_tokens}
    except Exception as e:
        return {"ran": False, "reason": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description="AMD hardware benchmark harness")
    ap.add_argument("--matmul", type=int, default=8192, help="matrix size")
    ap.add_argument("--iters", type=int, default=30, help="matmul iterations")
    ap.add_argument("--llm", type=str, default="", help="optional LLM prompt")
    ap.add_argument("--vllm", type=str, default="",
                    help="optional model id to benchmark with vLLM on ROCm")
    ap.add_argument("--vllm-prompts", type=int, default=16,
                    help="batch size for the vLLM throughput run")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    runs = []
    print("[bench] probing device...")
    dev = device_report()
    print(json.dumps(dev, indent=2))

    print(f"[bench] matmul {args.matmul}x{args.matmul} x{args.iters}...")
    mm = matmul_benchmark(args.matmul, args.iters)
    print(json.dumps(mm, indent=2))
    runs.append({"name": "fp16 matmul", **mm})

    if args.llm:
        print("[bench] LLM round-trip...")
        ll = llm_latency(args.llm)
        print(json.dumps(ll, indent=2))
        runs.append({"name": "llm round-trip", **ll})

    if args.vllm:
        print(f"[bench] vLLM throughput ({args.vllm}) on ROCm...")
        vl = vllm_throughput(args.vllm, prompts=args.vllm_prompts)
        print(json.dumps(vl, indent=2))
        runs.append({"name": "vLLM throughput", **vl})

    out = {
        "generated_utc": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "device": dev,
        "runs": runs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
