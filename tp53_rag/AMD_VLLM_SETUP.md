# Running Gemma on AMD Instinct with vLLM (Developer Cloud runbook)

This is the exact path to serve an **open Gemma model on an AMD Instinct MI300X**
via **vLLM on ROCm**, and point Precision Onco Africa at it. Doing this turns
four otherwise "roadmap" items into **real, demonstrable** features:

- **Continuous batching** (the "single batch tensor" — automatic in vLLM)
- **FP8 KV-cache compression** (`--kv-cache-dtype fp8`)
- **Speculative decoding / "Little Gemma"** (a 2B draft model + the 27B target)
- **Live `rocm-smi` GPU telemetry** (real numbers for the hardware widget + the
  autonomic self-healing demo)

...and it lets you truthfully say *"Gemma, served on AMD Instinct via vLLM,"* —
strengthening **both** the Track-3 (AMD) and the Gemma tracks at once.

> Honesty note: everything here runs on **real AMD hardware**. The GPU widgets
> and the self-healing demo must be recorded **on this instance** (where
> `rocm-smi` is real) — never simulate GPU telemetry on a non-AMD machine.

---

## Part A — Get an AMD GPU instance

1. Log in to the **AMD AI Developer Program** portal
   (developer.amd.com/ai-developer-program). This is the "Secure Site" in your
   AMD account.
2. Dashboard → **Member Perks** → **Request Cloud Credits** (if you haven't
   already). Fill the form (affiliation, use-case, a public GitHub/LinkedIn for
   verification); approval takes ~2–3 business days and the **$100 credit** +
   activation link arrive by email. *(You've already done this.)*
3. Open the **AMD Developer Cloud** console from that activation email. It is a
   DigitalOcean-powered GPU cloud.
4. **Billing → add a payment card first.** Even with $100 in credits, the
   **Create GPU Droplet** button stays greyed out until a card is on file
   (credits are spent first; the card only covers overage). *This is the exact
   blocker you hit before — a standard (not prepaid) card is required.*
5. Left sidebar → **GPU Droplets** → **Create GPU Droplet**.
   - Region: **ATL1 (Atlanta)** — most reliable MI300X availability.
   - GPU: **MI300X** (192 GB HBM3 — runs `gemma-2-27b-it` on a single card).
   - Image: pick a ROCm base (or plain Ubuntu — the vLLM ROCm *container* below
     brings its own ROCm userspace).
6. Create it, then **SSH in** using the connection details shown. Sanity-check:
   ```bash
   rocm-smi          # should list the MI300X
   amd-smi monitor   # newer equivalent
   ```

> Shortcut: lablab has a step-by-step "Host your first LLM on AMD GPU with vLLM"
> tutorial that mirrors Part B/C — worth following alongside this.

---

## Part B — Launch the ROCm vLLM container

Use the **prebuilt image** — do NOT build vLLM from source on ROCm.

```bash
docker pull rocm/vllm:latest          # check Docker Hub for the newest tag

docker run -it --rm \
  --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 16G \
  --security-opt seccomp=unconfined \
  -e HF_TOKEN=<your_huggingface_token> \
  -v $HOME/models:/models \
  rocm/vllm:latest
```

The `--device=/dev/kfd --device=/dev/dri --group-add video` flags are the ROCm
GPU passthrough — without them vLLM won't see the GPU.

---

## Part C — Serve Gemma + turn on the "wow" flags

Gemma is **gated**: accept the licence at huggingface.co/google/gemma-2-27b-it
and make sure `HF_TOKEN` is set, or the download 401s.

Baseline (already gives continuous batching + a clean instruct model):
```bash
vllm serve google/gemma-2-27b-it --host 0.0.0.0 --port 8000
```

Full "wow" configuration:
```bash
vllm serve google/gemma-2-27b-it \
  --host 0.0.0.0 --port 8000 \
  --kv-cache-dtype fp8 \                       # FP8 KV-cache compression
  --speculative-model google/gemma-2-2b-it \   # "Little Gemma" draft model
  --num-speculative-tokens 5                   # speculative decoding / MTP
```

> Flag syntax drifts between vLLM versions. If it errors, run `vllm serve --help`
> — newer builds fold speculation into a single `--speculative-config '{...}'`
> JSON argument. `--kv-cache-dtype fp8` is stable.

---

## Part D — Point the app at vLLM (no new code)

vLLM's server is **OpenAI-compatible**, and `FireworksBackend` is already a
generic OpenAI-compatible client. In `.env`:

```
INFERENCE_MODE=fireworks
FIREWORKS_BASE_URL=http://localhost:8000/v1     # or http://<instance-ip>:8000/v1
FIREWORKS_MODEL=google/gemma-2-27b-it
FIREWORKS_API_KEY=dummy                          # vLLM ignores unless --api-key set
```

- Run the Streamlit app **on the instance** → use `localhost:8000`.
- Run it on your **laptop** → SSH-tunnel so you don't expose the port publicly:
  ```bash
  ssh -L 8000:localhost:8000 <user>@<instance-ip>
  ```

---

## Part E — Real numbers + real telemetry

Once vLLM is serving:

```bash
# Real ROCm throughput → fills the last placeholder in AMD_SUBMISSION.md
python tools/benchmark_amd.py --vllm google/gemma-2-27b-it

# Live GPU stats for the hardware widget + autonomic self-healing demo
rocm-smi            # real utilisation + VRAM
```

Record the GPU widget and the "VRAM spike → self-heal" demo **here**, on the
real MI300X — that's the honest, judge-proof version.

---

## Watch-outs

- **Gemma is gated** — accept the HF licence + set `HF_TOKEN`.
- **Card required** even with credits (Part A step 4); use a standard card.
- **Credits expire ~30 days** from deposit, and the instance **bills while
  running** — spin up for the benchmark + demo recording, then **stop the
  droplet**.
- **Model size:** MI300X (192 GB) runs `gemma-2-27b-it` comfortably; drop to
  `gemma-2-9b-it` on smaller cards.
- **Image tag drift:** if `rocm/vllm:latest` misbehaves, pin the known-good tag
  from AMD's vLLM-on-ROCm docs.
