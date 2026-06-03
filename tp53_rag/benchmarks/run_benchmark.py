"""
TP53 Variant Curator — accuracy benchmark.

Runs the curated ground-truth variants through the existing VariantCurator and
scores its structured output against ClinVar/IARC reference classifications.

Pure offline: VariantCurator.classify() is rule-based, so no LLM/Ollama is
required. Opt-in: nothing here is imported by the live app.

Usage:
    python -m benchmarks.run_benchmark
    python benchmarks/run_benchmark.py --no-save
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from benchmarks.scoring import score_variant, aggregate

log = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = BENCH_DIR / "ground_truth.json"
RESULTS_DIR = BENCH_DIR / "results"


def load_ground_truth(path: Optional[Path] = None) -> Dict:
    """Load and lightly validate the ground-truth file.

    Returns a dict with 'metadata' and 'variants'. On any failure returns an
    empty-but-valid structure (never raises) so the runner degrades gracefully.
    """
    path = Path(path) if path else GROUND_TRUTH_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        variants = data.get("variants", [])
        if not isinstance(variants, list):
            raise ValueError("'variants' must be a list")
        return {"metadata": data.get("metadata", {}), "variants": variants}
    except Exception as e:
        log.error(f"Failed to load ground truth from {path}: {e}")
        return {"metadata": {"error": str(e)}, "variants": []}


def _default_classifier() -> Callable[[str], Dict]:
    """Return a callable mutation -> classification-dict using VariantCurator.

    Isolated so a failed import degrades to an informative stub instead of
    crashing the whole run.
    """
    try:
        from agents.variant_curator import VariantCurator
        curator = VariantCurator()

        def _classify(mutation: str) -> Dict:
            out = curator.classify(mutation)
            return (out or {}).get("classification", {})

        return _classify
    except Exception as e:  # pragma: no cover - import environment dependent
        log.error(f"VariantCurator unavailable: {e}")

        def _stub(mutation: str) -> Dict:
            return {}

        return _stub


def run_benchmark(classifier: Optional[Callable[[str], Dict]] = None,
                  ground_truth: Optional[Dict] = None) -> Dict:
    """Execute the benchmark.

    Args:
        classifier: mutation -> classification-dict. Defaults to VariantCurator.
                    Injectable so tests run without the real agent.
        ground_truth: pre-loaded ground-truth dict. Defaults to the JSON file.

    Returns:
        {"metrics": {...}, "results": [...], "metadata": {...}, "timestamp": ...}
        Always populated; never raises on a single bad variant.
    """
    gt = ground_truth if ground_truth is not None else load_ground_truth()
    variants: List[Dict] = gt.get("variants", [])
    classify = classifier or _default_classifier()

    results: List[Dict] = []
    for v in variants:
        mutation = v.get("mutation", "")
        try:
            predicted = classify(mutation) or {}
        except Exception as e:
            log.warning(f"Classifier failed on {mutation!r}: {e}")
            predicted = {}
        results.append(score_variant(predicted, v))

    return {
        "metrics": aggregate(results),
        "results": results,
        "metadata": gt.get("metadata", {}),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def render_markdown(report: Dict) -> str:
    """Render a human-readable markdown report. Always returns a non-empty string."""
    m = report.get("metrics", {})
    results = report.get("results", [])
    meta = report.get("metadata", {})

    lines = [
        "# TP53 Variant Curator — Benchmark Report",
        "",
        f"_Generated: {report.get('timestamp', '?')}_  ",
        f"_Source: {meta.get('description', 'ground-truth dataset')}_",
        "",
        "## Summary",
        "",
        f"- Variants evaluated: **{m.get('n', 0)}**",
        f"- Exact-match accuracy: **{m.get('exact_accuracy', 0):.0%}**",
        f"- Concordant (bucketed) accuracy: **{m.get('concordant_accuracy', 0):.0%}**",
        f"- IARC concordance: **{m.get('iarc_concordance', 0):.0%}** "
        f"(scored on {m.get('iarc_scored', 0)} variants)",
        f"- Pathogenic detection — precision **{m.get('precision', 0):.0%}**, "
        f"recall **{m.get('recall', 0):.0%}**, F1 **{m.get('f1', 0):.0%}**",
        f"- Confusion: TP={m.get('tp', 0)} FP={m.get('fp', 0)} "
        f"FN={m.get('fn', 0)} TN={m.get('tn', 0)}",
        "",
        "## Per-variant results",
        "",
        "| Variant | Expected | Predicted | Exact | Concordant | IARC (exp/pred) |",
        "|---|---|---|:---:|:---:|---|",
    ]
    for r in results:
        exact = "✅" if r.get("exact_match") else "❌"
        conc = "✅" if r.get("concordant") else "❌"
        iarc_m = r.get("iarc_match")
        iarc_cell = (
            f"{r.get('expected_iarc') or '—'} / {r.get('predicted_iarc') or '—'}"
            + (" ✅" if iarc_m else (" ❌" if iarc_m is False else ""))
        )
        lines.append(
            f"| {r.get('mutation','?')} | {r.get('expected_significance','?')} | "
            f"{r.get('predicted_significance','?')} | {exact} | {conc} | {iarc_cell} |"
        )

    disclaimer = meta.get("disclaimer")
    if disclaimer:
        lines += ["", "---", f"> {disclaimer}"]
    return "\n".join(lines)


def save_report(report: Dict, results_dir: Optional[Path] = None) -> Dict[str, str]:
    """Persist JSON + markdown reports. Returns the written paths."""
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"benchmark_{stamp}.json"
    md_path = results_dir / f"benchmark_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="TP53 variant curator benchmark")
    parser.add_argument("--no-save", action="store_true",
                        help="Print the report but do not write files.")
    args = parser.parse_args(argv)

    report = run_benchmark()
    md = render_markdown(report)
    # Report goes to the logger/stdout via the markdown body (no bare print of
    # data; use the logging channel for the machine-readable summary).
    log.info("Benchmark complete: %s", json.dumps(report["metrics"]))
    if args.no_save:
        log.info("\n%s", md)
    else:
        paths = save_report(report)
        log.info("Wrote report: %s", paths["markdown"])
        log.info("\n%s", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
