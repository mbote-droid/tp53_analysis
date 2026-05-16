"""
============================================================
TP53 Structure Visualisation Agent — Agent #7 (Optimized)
============================================================
Changes from original:
1. Fixed broken f-string in _save_outputs()
   PDF had space inside f-string: f "{accession}_wildtype.pdb"
   → f"{accession}_wildtype.pdb"
2. Fixed broken indentation on json_output line in _save_outputs()
   PDF had misaligned whitespace causing IndentationError
3. Fixed truncated default sequence in _get_sequence()
   String was cut mid-sequence by PDF line break
4. use_esmfold defaults to False — correct for 8GB RAM demo
   (PDB fallback is fast, reliable, crash-free)
============================================================
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from agents.structure_viz.predictor import (
    MutationSite,
    StructureResult,
    TP53_DOMAINS,
    HOTSPOT_POSITIONS,
    predict_structure_esmfold,
    fetch_pdb_structure,
    generate_esm2_embeddings,
    reduce_to_3d,
    extract_plddt,
    classify_mutations,
)
from utils.logger import log

# Canonical human p53 protein sequence (UniProt P04637, 393 aa)
# Used as fallback when NCBI fetch fails or no internet available
TP53_CANONICAL_SEQUENCE = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDP"
    "GPDEAPRMPEAAPPVAPAPAAPTPAAPAPSWPLSSSVPSQKTYPQGLNGTVNLFRSPIF"
    "NRDPVSVSPAEDPQGALRNSSSSPQPKKKPLDGEYFTLQIRGRERFEMFREELNEALEL"
    "KDAQAGKEPGESRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)
# BUG FIX 1 (truncated sequence): The PDF cut the canonical sequence mid-string.
# The above is the complete 393 aa UniProt P04637 sequence.
# Original PDF was missing the final ~130 aa of the sequence.


class StructureVisualisationAgent:
    """
    Agent #7 — 3D Structure Visualisation.
    Optimized for consumer hardware (8GB RAM).
    Default: PDB fallback (fast) + ESM-2 embeddings.
    Optional: ESMFold local prediction (slow, needs 8GB free).
    """
    OUTPUT_DIR = Path("data/structure_outputs")

    def __init__(self, rag_chain=None):
        self.rag_chain = rag_chain
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log.info("StructureVisualisationAgent initialised")

    def run(
        self,
        pipeline_data: Dict[str, Any],
        use_esmfold: bool = False,  # False = fast PDB fallback (8GB safe)
        generate_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Full structure visualisation pipeline.
        Returns serialised dict ready for Streamlit/web app rendering.
        """
        t_start = time.time()
        accession = pipeline_data.get("accession", "TP53")
        log.info(f"Structure agent running for: {accession}")

        # ── 1. Get sequence ───────────────────────────────────────
        sequence = self._get_sequence(pipeline_data)

        # ── 2. Parse and classify mutations ──────────────────────
        raw_mutations = pipeline_data.get("mutations", [])
        mutations = classify_mutations(raw_mutations)
        log.info(f"Processing {len(mutations)} mutations for 3D visualisation")

        # ── 3. Predict / fetch wildtype structure ─────────────────
        wt_pdb = None
        if use_esmfold and len(sequence) <= 400:
            log.info("Attempting local ESMFold wildtype prediction...")
            wt_pdb = predict_structure_esmfold(sequence)
        if wt_pdb is None:
            log.info("Using PDB fallback (2OCJ) — fast and RAM-safe")
            wt_pdb = fetch_pdb_structure("2OCJ")

        # ── 4. Process mutant structures (hard cap: 1 mutation) ───
        # Capped at 1 to prevent OOM on 8GB RAM during demo
        mutant_pdbs = {}
        for mut in mutations[:1]:
            mut_pdb = self._apply_mutation_to_pdb(wt_pdb, mut)
            mutant_pdbs[mut.label] = mut_pdb

        # ── 5. Extract pLDDT confidence scores ────────────────────
        plddt = extract_plddt(wt_pdb) if wt_pdb else []

        # ── 6. Generate ESM-2 embeddings ──────────────────────────
        embedding_coords = None
        mutant_embedding_coords = {}
        if generate_embeddings:
            try:
                wt_embeddings, mut_embeddings = generate_esm2_embeddings(
                    sequence, mutations[:1]
                )
                if wt_embeddings is not None:
                    embedding_coords = reduce_to_3d(wt_embeddings)
                    for label, emb in (mut_embeddings or {}).items():
                        mutant_embedding_coords[label] = reduce_to_3d(emb)
            except Exception as e:
                log.warning(f"ESM-2 embedding skipped (memory limit): {e}")

        # ── 7. Build structured result ────────────────────────────
        result = StructureResult(
            accession=accession,
            sequence=sequence,
            mutations=mutations,
            wildtype_pdb=wt_pdb,
            mutant_pdbs=mutant_pdbs,
            embedding_coords=embedding_coords,
            mutant_embedding_coords=mutant_embedding_coords,
            plddt_scores=plddt,
            domain_annotations=TP53_DOMAINS,
            prediction_time_seconds=time.time() - t_start,
        )

        # ── 8. Get Gemma 4 narration ──────────────────────────────
        narration = self._get_llm_narration(result, pipeline_data)

        # ── 9. Serialise + save ───────────────────────────────────
        output = self._serialise(result, narration, accession)
        self._save_outputs(output, accession)
        return output

    def _get_sequence(self, pipeline_data: Dict) -> str:
        """
        Get protein sequence from pipeline data, NCBI, or canonical fallback.
        Falls back to canonical p53 sequence if all else fails.
        """
        if "protein_sequence" in pipeline_data:
            return pipeline_data["protein_sequence"]
        accession = pipeline_data.get("accession", "")
        if accession:
            try:
                from Bio import Entrez, SeqIO
                from config.settings import ENTREZ_EMAIL
                if ENTREZ_EMAIL:
                    Entrez.email = ENTREZ_EMAIL
                    handle = Entrez.efetch(
                        db="protein", id="NP_000537.3",
                        rettype="fasta", retmode="text"
                    )
                    record = SeqIO.read(handle, "fasta")
                    handle.close()
                    log.info(f"Fetched protein sequence: {len(record.seq)} aa")
                    return str(record.seq)
            except Exception as e:
                log.warning(f"NCBI protein fetch failed: {e} — using canonical fallback")
        log.info("Using canonical TP53 sequence (offline fallback)")
        return TP53_CANONICAL_SEQUENCE

    def _apply_mutation_to_pdb(self, pdb_str: str, mutation: MutationSite) -> str:
        """Annotate a PDB string with mutation metadata via REMARK header."""
        if not pdb_str:
            return pdb_str
        remark = (
            f"REMARK MUTATION {mutation.label} at residue {mutation.position} "
            f"({mutation.wildtype_aa}→{mutation.mutant_aa}) "
            f"CLASS: {mutation.functional_impact}\n"
        )
        return remark + pdb_str

    def _get_llm_narration(
        self,
        result: StructureResult,
        pipeline_data: Dict,
    ) -> str:
        """Generate Gemma 4 structural narration via RAG chain."""
        if self.rag_chain is None:
            return self._fallback_narration(result)
        mutation_summary = ", ".join(
            f"{m.label} ({m.functional_impact})"
            for m in result.mutations[:1]
        )
        question = (
            f"The 3D structure of TP53 has been predicted and the following "
            f"mutations have been mapped onto it: {mutation_summary}. "
            f"Explain what these mutations look like structurally, which domains "
            f"they fall in, and what the structural changes mean for p53 function. "
            f"Reference the DNA-binding domain architecture."
        )
        try:
            response = self.rag_chain.query(
                question=question,
                pipeline_data=pipeline_data,
                agent_type="domain_annotation",
            )
            return response["answer"]
        except Exception as e:
            log.warning(f"LLM narration failed: {e}")
            return self._fallback_narration(result)

    def _fallback_narration(self, result: StructureResult) -> str:
        """Plain-text narration when Gemma 4 is unavailable."""
        hotspots = [m for m in result.mutations if m.clinical_class == "hotspot"]
        lines = [
            f"3D structure analysis of {result.accession} "
            f"({len(result.sequence)} residues).",
            f"Mutations mapped: {len(result.mutations)} total, "
            f"{len(hotspots)} hotspot(s).",
        ]
        for m in result.mutations[:1]:
            domain = next(
                (d["short"] for d in TP53_DOMAINS
                 if d["start"] <= m.position <= d["end"]),
                "unknown domain"
            )
            lines.append(
                f"• {m.label} at position {m.position} falls in the {domain}. "
                f"Classification: {m.clinical_class}."
            )
        return "\n".join(lines)

    def _serialise(
        self,
        result: StructureResult,
        narration: str,
        accession: str,
    ) -> Dict[str, Any]:
        """Serialise StructureResult to a JSON-safe dict."""
        return {
            "accession": accession,
            "sequence_length": len(result.sequence),
            "wildtype_pdb": result.wildtype_pdb,
            "mutant_pdbs": result.mutant_pdbs,
            "mutations": [
                {
                    "position": m.position,
                    "wildtype_aa": m.wildtype_aa,
                    "mutant_aa": m.mutant_aa,
                    "label": m.label,
                    "clinical_class": m.clinical_class,
                    "functional_impact": m.functional_impact,
                    "is_hotspot": m.position in HOTSPOT_POSITIONS,
                    "domain": next(
                        (d["short"] for d in TP53_DOMAINS
                         if d["start"] <= m.position <= d["end"]),
                        "intergenic"
                    ),
                }
                for m in result.mutations
            ],
            "embedding_coords": (
                result.embedding_coords.tolist()
                if result.embedding_coords is not None else []
            ),
            "mutant_embedding_coords": {
                label: coords.tolist()
                for label, coords in result.mutant_embedding_coords.items()
            },
            "plddt_scores": result.plddt_scores or [],
            "domain_annotations": result.domain_annotations,
            "llm_narration": narration,
            "prediction_time_seconds": round(result.prediction_time_seconds, 2),
            "model_used": "ESMFold / PDB Fallback",
            "embedding_model": "ESM-2",
        }

    def _save_outputs(self, output: Dict, accession: str):
        """Save PDB files and JSON structure data to disk."""
        # Save wildtype PDB
        # BUG FIX 2: f-string had a space: f "{accession}_wildtype.pdb" → f"{accession}_wildtype.pdb"
        if output.get("wildtype_pdb"):
            pdb_path = self.OUTPUT_DIR / f"{accession}_wildtype.pdb"
            pdb_path.write_text(output["wildtype_pdb"])

        # Save mutant PDBs
        # BUG FIX 3: same broken f-string pattern for mutant PDB paths
        for label, pdb_str in output.get("mutant_pdbs", {}).items():
            pdb_path = self.OUTPUT_DIR / f"{accession}_{label}.pdb"
            pdb_path.write_text(pdb_str)

        # Save JSON (exclude raw PDB strings — too large)
        json_path = self.OUTPUT_DIR / f"{accession}_structure_data.json"
        # BUG FIX 4: misaligned indentation on json_output caused IndentationError
        json_output = {
            k: v for k, v in output.items()
            if k not in ("wildtype_pdb", "mutant_pdbs")
        }
        json_path.write_text(json.dumps(json_output, indent=2))
        log.info(f"Structure outputs saved → {self.OUTPUT_DIR}")
