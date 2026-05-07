# Debugging Journal — TP53 Bioinformatics Pipeline

> This document records every real bug encountered during development,
> how each was diagnosed, and how it was fixed. It is included
> deliberately — debugging is half of bioinformatics work, and
> understanding failure modes is as important as writing features.

---

## Bug 1 — HTTP 400 Bad Request on Every NCBI Fetch

**Symptom**
```
[WARNING] Attempt 1/3: HTTP Error 400. Retrying...
[WARNING] Attempt 2/3: HTTP Error 400. Retrying...
[WARNING] Attempt 3/3: HTTP Error 400. Retrying...
[ERROR] Failed to fetch accession 'NM_000546' after 3 attempts: HTTP Error 400: Bad Request
```
Every fetch attempt failed regardless of accession ID or email.
Retrying made no difference.

**Investigation**
Confirmed the email and accession were correctly formatted.
Isolated `fetch_sequence()` and inspected the parameters being
passed to `Entrez.efetch()`. Found that `Entrez.api_key` was
being set unconditionally from the environment variable.

**Root Cause**
```python
# BEFORE — the bug
Entrez.api_key = os.environ.get("NCBI_API_KEY", "").strip()
```
When `NCBI_API_KEY` is not set, `os.environ.get()` returns an
empty string `""`. BioPython then appends `&api_key=` (blank)
to every NCBI request URL. NCBI's server rejects malformed
parameters with HTTP 400 Bad Request.

**Fix**
```python
# AFTER — only assign when a real value exists
_api_key = os.environ.get("NCBI_API_KEY", "").strip()
if _api_key:
    Entrez.api_key = _api_key
```

**Lesson Learned**
Never assign environment variables directly without checking
for empty strings. `os.environ.get("KEY", "")` and
`os.environ.get("KEY")` behave very differently downstream.
An empty string is not the same as a missing value.

---

## Bug 2 — Valid NCBI Sequences Rejected as Invalid DNA

**Symptom**
Pipeline logged warnings about non-standard nucleotides on
sequences fetched directly from NCBI — sequences that are
by definition valid.

**Root Cause**
```python
# BEFORE — too narrow
valid_chars = set("ACGTN-")
```
Real NCBI mRNA records use the full IUPAC nucleotide ambiguity
alphabet. Characters like `R, Y, S, W, K, M, B, D, H, V` are
all legitimate and common in GenBank records. NM_000546 itself
contains several of these.

**Fix**
```python
# AFTER — full IUPAC alphabet
valid_chars = set("ACGTNRYSWKMBDHV-")
```

**Lesson Learned**
Always validate against the actual standard, not an assumption
of what the data looks like. IUPAC ambiguity codes exist for
a biological reason — they represent positions where the exact
base is uncertain or variable across a population.

---

## Bug 3 — Type Mismatch: str + Seq Object

**Symptom**
Silent incorrect results during alignment and mutation
detection. No crash — wrong output.

**Root Cause**
```python
# BEFORE — mixing Python str and BioPython Seq
mutant_dna = "A" * 50 + dna_seq[50:1000]
```
The left side is a plain Python `str`, the right side is a
BioPython `Seq` object. Python concatenates them without
error but downstream BioPython functions behave inconsistently
with mixed types — some silently produce wrong results.

**Fix**
```python
# AFTER — both sides are Seq objects
mutant_dna = Seq("A" * n) + dna_seq[n:compare_len]
```

**Lesson Learned**
BioPython's `Seq` type is not interchangeable with Python `str`
even though many operations appear to work. Always use `Seq()`
explicitly when constructing sequences for BioPython functions.

---

## Bug 4 — plt.show() Blocking Execution on Headless Servers

**Symptom**
Pipeline hung indefinitely when run on a server or in CI.
No output was produced after the plotting step.

**Root Cause**
```python
# BEFORE
plt.show()
```
`plt.show()` is an interactive call that opens a GUI window
and blocks execution until the window is closed. On headless
servers (no display), Linux CI runners, or remote machines,
there is no GUI — the call blocks forever.

**Fix**
```python
# AFTER
matplotlib.use("Agg")  # non-interactive backend at top of file
plt.savefig("results/tp53_analysis.png", dpi=150)
plt.close()
```
Setting the `Agg` backend before importing pyplot tells
Matplotlib to render to file rather than screen — works
on any environment.

**Lesson Learned**
Never use `plt.show()` in a pipeline or script intended to
run non-interactively. Always save to file and close the
figure to free memory.

---

## Bug 5 — Logger is None When Functions Are Imported in Tests

**Symptom**
```
AttributeError: 'NoneType' object has no attribute 'error'
```
42 out of 63 tests failed immediately on first test run.

**Root Cause**
```python
# In main_tp53_analysis.py
logger = None  # set at module level

def main():
    logger = setup_logging()  # only initialised here
```
The pipeline initialises `logger` inside `main()`. When pytest
imports individual functions directly, `main()` never runs,
so `logger` remains `None`. Every function that calls
`logger.error()` or `logger.info()` then crashes.

**Fix**
Added logger initialisation at the top of the test file:
```python
import main_tp53_analysis as _pipeline_module
import logging as _logging
_pipeline_module.logger = _logging.getLogger("TP53Pipeline")
_pipeline_module.logger.setLevel(_logging.DEBUG)
_pipeline_module.logger.addHandler(_logging.NullHandler())
```
`NullHandler` means the logger is valid but produces no
output during tests — clean test results.

**Lesson Learned**
Module-level globals that depend on a function being called
first are a testing hazard. A better long-term pattern is to
initialise the logger at module level unconditionally, and
let the caller configure its handlers.

---

## Bug 6 — Empty Sequence Passes Validation

**Symptom**
```
FAILED TestValidateSequence::test_empty_sequence_rejected
AssertionError: True is not false
```
`validate_sequence(Seq(""), "DNA")` returned `True` instead
of `False`.

**Root Cause**
```python
return all(c in valid_chars for c in seq_str)
```
Python's `all()` on an empty iterable always returns `True`
by definition — there are no elements to fail the check.
An empty string passes vacuously.

**Fix**
```python
# Added explicit empty check before the all() call
if not seq_str:
    return False
```

**Lesson Learned**
`all()` on an empty collection is a classic Python gotcha.
Any validation function that uses `all()` must explicitly
guard against empty input first.

---

## Bug 7 — Incorrect Biological Assumption in Test

**Symptom**
```
FAILED TestAnalyzeProtein::test_empty_protein_if_immediate_stop
AssertionError: 1 != 0
```

**Root Cause**
The test assumed `ATGTAA` would produce an empty protein:
```python
protein = analyze_protein(Seq("ATGTAA"))
self.assertEqual(len(protein), 0)  # WRONG
```
`ATG` codes for Methionine (Met). `TAA` is the stop codon.
With `to_stop=True`, translation stops *before* including
the stop codon but *after* translating `ATG` → `M`.
The result is a protein of length 1, not 0.

**Fix**
```python
# AFTER — corrected biological expectation
self.assertEqual(len(protein), 1)
self.assertEqual(str(protein), "M")
```

**Lesson Learned**
`to_stop=True` in BioPython means "stop at the stop codon
and exclude it" — not "exclude everything including the
last codon before it." Always verify biological assumptions
against a known sequence before writing the test assertion.

---

## Bug 8 — export_mutations_csv() Early Exit on Empty List

**Symptom**
```
FAILED TestFileOperations::test_results_directory_created_automatically
AssertionError: False is not true
```
The results directory was not being created.

**Root Cause**
`export_mutations_csv([])` returns early with a warning when
passed an empty list — so `os.makedirs()` was never reached
and the directory was never created.

**Fix**
Updated the test to pass a real mutation entry so the
function executes fully:
```python
export_mutations_csv(
    [{"position": 1, "original": "A", "mutant": "G"}],
    filepath=fpath
)
```

**Lesson Learned**
Test the function's full execution path, not just the happy
path. Early-exit conditions must be tested separately from
the main behaviour being verified.

---

## Summary

| # | Bug | Type | Impact |
|---|-----|------|--------|
| 1 | Empty api_key causes HTTP 400 | Environment variable handling | Critical — pipeline unusable |
| 2 | Narrow IUPAC alphabet | Incorrect standard | High — rejects valid sequences |
| 3 | str + Seq type mismatch | Type safety | High — silent wrong results |
| 4 | plt.show() blocks on server | Platform assumption | High — hangs in CI/server |
| 5 | Logger None in tests | Architecture | Medium — 42 test failures |
| 6 | all() on empty iterable | Python gotcha | Medium — false validation pass |
| 7 | Wrong biological assumption | Domain knowledge | Low — test only |
| 8 | Early exit skips makedirs | Test coverage gap | Low — test only |

---

*Written by Samuel Mbote — General Surgery Resident & Bioinformatics Developer*