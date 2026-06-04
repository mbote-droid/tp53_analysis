# Dependency locking

This project ships two dependency files:

| File | Purpose |
|---|---|
| `requirements.txt` | **Flexible** install with bounded ranges (`>=X,<Y`). Default — what Streamlit Cloud reads. Lets the resolver pick compatible patch versions while blocking the bleeding-edge majors that previously broke the deploy. |
| `requirements.lock` | **Pinned** exact `==` versions of the direct dependencies, taken from the last known-good Python-3.11 cloud build. Use for reproducible / audited deploys. |

## Why a lock matters here

Earlier deploys broke repeatedly because unbounded `>=` specs pulled bleeding-edge
majors (langchain 1.x, pandas 3.x, protobuf 7.x) that were incompatible with the
code. `requirements.txt` now caps majors; `requirements.lock` goes further and
pins exact versions so a build is byte-for-byte reproducible.

## Using the lock

- **Reproducible cloud deploy:** point the Streamlit app at `requirements.lock`
  (App → Settings), or copy it over `requirements.txt` before deploying.
- **Local reproducible install:**
  ```bash
  pip install -r requirements.lock
  ```

## Regenerating a full transitive lock (recommended before launch)

The committed lock pins **direct** dependencies. For a complete, exact lock of
the entire transitive tree, generate it **on the deploy target** (Python 3.11,
cloud-safe block only — no torch / sentence-transformers / whisper):

```bash
# in a clean Python 3.11 venv
pip install -r requirements.txt
pip freeze > requirements.lock
```

Or with uv / pip-tools:

```bash
uv pip compile requirements.txt -o requirements.lock
# or
pip-compile requirements.txt -o requirements.lock
```

> Note: `protobuf` is intentionally left as a bound (`>=4.25,<6`) rather than an
> exact pin — it must stay below 6 to avoid breaking chromadb's opentelemetry
> `_pb2` import, but the exact patch is environment-resolved. Regeneration on the
> deploy env will fix it to a concrete value.
