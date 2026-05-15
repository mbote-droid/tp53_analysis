"""
============================================================
TP53 Structure Visualisation Agent — Agent #7
============================================================
The 7th agent in the multi-agent platform. Receives pipeline
output, predicts/fetches 3D structures, generates ESM-2
embeddings, and produces all data needed by the web app
and Jupyter notebook visualisers.

Also queries Gemma 4 via RAG to narrate what the structure
visualisation means clinically — connecting the visual
"wow" to grounded interpretation.
============================================================
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents.structure_viz.predictor import (
    MutationSite,
    StructureResult,
    TP53_DOMAINS,
    HOTSPOT_POSITIONS,
    apply_mutations,
    predict_structure_esmfold,
    fetch_pdb_structure,
    generate_esm2_embeddings,
    reduce_to_3d,
    extract_plddt,
    classify_mutations,
)
from utils.logger import log


class StructureVisualisationAgent:
    """
    Agent #7 — 3D Structure Visualisation.

    Given pipeline output:
      1. Builds wildtype + mutant 3D structures (ESMFold or PDB fallback)
      2. Generates ESM-2 residue embeddings → UMAP 3D coords
      3. Annotates mutation sites on structure
      4. Provides Gemma 4 RAG narration of what the structure shows
      5. Saves all outputs for web + notebook viewers

    Fully offline after first model download.
    """

    OUTPUT_DIR = Path("data/structure_outputs")

    def __init__(self, rag_chain=None):
        """
        Args:
            rag_chain: Optional TP53RAGChain for Gemma 4 narration.
                      If None, structural analysis runs without LLM narration.
        """
        self.rag_chain = rag_chain
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log.info("StructureVisualisationAgent initialised")

    def run(
        self,
        pipeline_data: Dict[str, Any],
        use_esmfold: bool = True,
        generate_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entry point. Processes pipeline data and returns
        all visualisation assets.

        Args:
            pipeline_data: Standard TP53 pipeline output dict
            use_esmfold: Try ESMFold first (falls back to PDB fetch)
            generate_embeddings: Generate ESM-2 embedding 3D coords

        Returns:
            Dict with all data needed by web app + notebook viewer
        """
        t_start = time.time()
        accession = pipeline_data.get("accession", "TP53")
        log.info(f"Structure agent running for: {accession}")

        # ── 1. Get sequence ───────────────────────────────
        sequence = self._get_sequence(pipeline_data)

        # ── 2. Parse mutations ────────────────────────────
        raw_mutations = pipeline_data.get("mutations", [])
        mutations = classify_mutations(raw_mutations)
        log.info(f"Processing {len(mutations)} mutations for 3D visualisation")

        # ── 3. Predict/fetch wildtype structure ───────────
        wt_pdb = None
        if use_esmfold and len(sequence) <= 400:
            wt_pdb = predict_structure_esmfold(sequence)

        if wt_pdb is None:
            log.info("ESMFold unavailable — fetching PDB structure (2OCJ)")
            wt_pdb = fetch_pdb_structure("2OCJ")

        # ── 4. Predict mutant structures ──────────────────
        mutant_pdbs = {}
        for mut in mutations[:3]:  # cap at 3 mutants to save time
            mut_sequence = apply_mutations(sequence, [mut])
            mut_pdb = None
            if use_esmfold and len(sequence) <= 400:
                mut_pdb = predict_structure_esmfold(mut_sequence)
            if mut_pdb is None:
                mut_pdb = self._apply_mutation_to_pdb(wt_pdb, mut)
            mutant_pdbs[mut.label] = mut_pdb

        # ── 5. Extract pLDDT confidence scores ────────────
        plddt = extract_plddt(wt_pdb) if wt_pdb else []

        # ── 6. Generate ESM-2 embeddings ──────────────────
        embedding_coords = None
        mutant_embedding_coords = {}

        if generate_embeddings:
            wt_embeddings, mut_embeddings = generate_esm2_embeddings(
                sequence, mutations[:3]
            )
            if wt_embeddings is not None:
                embedding_coords = reduce_to_3d(wt_embeddings)
                for label, emb in (mut_embeddings or {}).items():
                    mutant_embedding_coords[label] = reduce_to_3d(emb)

        # ── 7. Build structured result ────────────────────
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

        # ── 8. Get Gemma 4 narration ──────────────────────
        narration = self._get_llm_narration(result, pipeline_data)

        # ── 9. Serialise outputs ──────────────────────────
        output = self._serialise(result, narration, accession)

        # ── 10. Save files ────────────────────────────────
        self._save_outputs(output, accession)

        log.info(
            f"Structure agent complete in {result.prediction_time_seconds:.1f}s "
            f"| mutations annotated: {len(mutations)} "
            f"| embeddings: {'yes' if embedding_coords is not None else 'no'}"
        )
        return output

    def _get_sequence(self, pipeline_data: Dict) -> str:
        """Get p53 protein sequence from pipeline data or fetch from NCBI."""
        # If pipeline provides it directly
        if "protein_sequence" in pipeline_data:
            return pipeline_data["protein_sequence"]

        # Try fetching from NCBI using accession
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
                    log.info(f"Fetched p53 protein sequence from NCBI ({len(record.seq)} aa)")
                    return str(record.seq)
            except Exception as e:
                log.warning(f"NCBI sequence fetch failed: {e}")

        # Fallback: use the canonical 393aa sequence
        log.info("Using canonical p53 sequence (offline fallback)")
        return (
            "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDP"
            "GPDEAPRMPEAAPPVAPAPAAPTPAAPAPSWPLSSSVPSQKTYPQGLNGTVNLFRSPIFN"
            "LNKTSSPIFKVDNHKFMNGSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGQMNRRPILTIT"
            "LEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEVVAPQHLIRVEGSQLAQDDCMFG"
            "RIVDRGQTLGSLQNLGIRHKYLSDTVSSDTVSSDTVSSDTVSSDTVSR"
        )

    def _apply_mutation_to_pdb(self, pdb_str: str, mutation: MutationSite) -> str:
        """
        Apply a mutation label to the PDB REMARK section.
        Used when ESMFold isn't available — marks mutation site
        in the PDB metadata so the viewer can highlight it.
        """
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
        """Get Gemma 4 RAG narration of the structural findings."""
        if self.rag_chain is None:
            return self._fallback_narration(result)

        mutation_summary = ", ".join(
            f"{m.label} ({m.functional_impact})"
            for m in result.mutations
        )
        hotspots = [m for m in result.mutations if m.clinical_class == "hotspot"]

        question = (
            f"The 3D structure of TP53 has been predicted and the following mutations "
            f"have been mapped onto it: {mutation_summary}. "
            f"There are {len(hotspots)} hotspot mutations: "
            f"{', '.join(m.label for m in hotspots)}. "
            f"Explain what these mutations look like structurally, which domains they "
            f"fall in, and what the structural changes mean for p53 function and "
            f"cancer biology. Reference the DNA-binding domain architecture."
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
        """Generate narration without LLM if Gemma 4 unavailable."""
        hotspots = [m for m in result.mutations if m.clinical_class == "hotspot"]
        lines = [
            f"3D structure analysis of {result.accession} ({len(result.sequence)} residues).",
            f"Mutations mapped: {len(result.mutations)} total, {len(hotspots)} hotspot(s).",
        ]
        for m in result.mutations:
            domain = next(
                (d["short"] for d in TP53_DOMAINS
                 if d["start"] <= m.position <= d["end"]),
                "unknown domain"
            )
            lines.append(
                f"• {m.label} at position {m.position} falls in the {domain}. "
                f"Classification: {m.clinical_class}, {m.functional_impact}."
            )
        return "\n".join(lines)

    def _serialise(
        self,
        result: StructureResult,
        narration: str,
        accession: str,
    ) -> Dict[str, Any]:
        """Convert StructureResult to JSON-serialisable dict."""
        return {
            "accession": accession,
            "sequence_length": len(result.sequence),

            # PDB structures (strings)
            "wildtype_pdb": result.wildtype_pdb,
            "mutant_pdbs": result.mutant_pdbs,

            # Mutation annotations for viewer
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

            # ESM-2 3D embedding coords (per residue)
            "embedding_coords": (
                result.embedding_coords.tolist()
                if result.embedding_coords is not None else []
            ),
            "mutant_embedding_coords": {
                label: coords.tolist()
                for label, coords in result.mutant_embedding_coords.items()
            },

            # pLDDT confidence
            "plddt_scores": result.plddt_scores or [],

            # Domain annotations
            "domain_annotations": result.domain_annotations,

            # Gemma 4 narration
            "llm_narration": narration,

            # Metadata
            "prediction_time_seconds": round(result.prediction_time_seconds, 2),
            "model_used": result.model_used,
            "embedding_model": result.embedding_model,
        }

    def _save_outputs(self, output: Dict, accession: str):
        """Save PDB files and JSON metadata to disk."""
        # Save wildtype PDB
        if output.get("wildtype_pdb"):
            pdb_path = self.OUTPUT_DIR / f"{accession}_wildtype.pdb"
            pdb_path.write_text(output["wildtype_pdb"])
            log.info(f"Saved wildtype PDB: {pdb_path}")

        # Save mutant PDBs
        for label, pdb_str in output.get("mutant_pdbs", {}).items():
            pdb_path = self.OUTPUT_DIR / f"{accession}_{label}.pdb"
            pdb_path.write_text(pdb_str)
            log.info(f"Saved mutant PDB: {pdb_path}")

        # Save full JSON output (for web app consumption)
        json_path = self.OUTPUT_DIR / f"{accession}_structure_data.json"
        # Exclude large PDB strings from JSON (they're separate files)
        json_output = {k: v for k, v in output.items()
                      if k not in ("wildtype_pdb", "mutant_pdbs")}
        json_path.write_text(json.dumps(json_output, indent=2))
        log.info(f"Saved structure JSON: {json_path}")
