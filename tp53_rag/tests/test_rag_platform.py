"""
============================================================
TP53 RAG Platform - Test Suite
============================================================
Tests for the ingestion pipeline, vector store, RAG chain,
and multi-agent dispatcher.

Run: pytest tests/ -v
     pytest tests/ --cov=. --cov-report=html
============================================================
"""

import socket

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from knowledge_base.ingestion import TP53DocumentIngester, CURATED_TP53_KNOWLEDGE
from agents.rag_chain import IntentRouter


def _has_network(host: str = "api-inference.huggingface.co",
                 port: int = 443, timeout: float = 3.0) -> bool:
    """True if an embeddings backend is reachable. Integration tests that build
    a real vector store need this; they skip (not fail) when offline."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _has_network(), reason="requires network access to an embeddings backend"
)


class TestCuratedKnowledge:
    """Validate the curated TP53 knowledge base content."""

    def test_curated_knowledge_not_empty(self):
        assert len(CURATED_TP53_KNOWLEDGE) > 0

    def test_curated_knowledge_has_required_fields(self):
        for item in CURATED_TP53_KNOWLEDGE:
            assert "content" in item
            assert "metadata" in item
            assert len(item["content"].strip()) > 50

    def test_curated_covers_critical_categories(self):
        categories = {item["metadata"]["category"] for item in CURATED_TP53_KNOWLEDGE}
        required = {"mutations", "clinical", "protein_domains", "gene_overview"}
        assert required.issubset(categories)

    def test_hotspot_mutations_present(self):
        all_content = " ".join(item["content"] for item in CURATED_TP53_KNOWLEDGE)
        hotspots = ["R175H", "R248W", "R248Q", "R273H", "R273C", "G245S"]
        for hotspot in hotspots:
            assert hotspot in all_content, f"Hotspot {hotspot} missing from knowledge base"

    def test_clinical_syndromes_covered(self):
        all_content = " ".join(item["content"] for item in CURATED_TP53_KNOWLEDGE)
        assert "Li-Fraumeni" in all_content
        assert "MDM2" in all_content


class TestIngester:
    """Test the document ingestion pipeline."""

    def test_load_curated_returns_documents(self):
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_load_curated_offline_flag(self):
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        for doc in docs:
            assert doc.metadata.get("offline_available") is True

    def test_chunk_documents(self):
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        chunks = ingester.chunk_documents(docs)
        # Chunks should be >= docs (each doc may split into multiple chunks)
        assert len(chunks) >= len(docs)

    def test_chunk_size_respected(self):
        from config.settings import CHUNK_SIZE
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        chunks = ingester.chunk_documents(docs)
        oversized = [c for c in chunks if len(c.page_content) > CHUNK_SIZE * 1.2]
        assert len(oversized) == 0, f"{len(oversized)} chunks exceed CHUNK_SIZE"

    def test_user_documents_empty_dir(self, tmp_path):
        ingester = TP53DocumentIngester()
        docs = ingester.load_user_documents(directory=tmp_path)
        assert docs == []


class TestIntentRouter:
    """Test the query routing logic."""

    def setup_method(self):
        self.router = IntentRouter()

    def test_mutation_keywords(self):
        queries = [
            "What is the impact of this mutation?",
            "Analyse the detected variants",
            "Tell me about the SNV at position 524",
        ]
        for q in queries:
            assert self.router.route(q) == "mutation_analysis", f"Failed for: {q}"

    def test_clinical_keywords(self):
        queries = [
            "What is the clinical significance?",
            "Is this pathogenic or benign?",
            "What cancer types are associated?",
        ]
        for q in queries:
            assert self.router.route(q) == "clinical_interpretation", f"Failed for: {q}"

    def test_phylogenetic_keywords(self):
        queries = [
            "Interpret the phylogenetic tree",
            "How conserved is this across species?",
            "What does the evolutionary analysis show?",
        ]
        for q in queries:
            assert self.router.route(q) == "phylogenetic_analysis", f"Failed for: {q}"

    def test_orf_keywords(self):
        result = self.router.route("Interpret the open reading frames found")
        assert result == "orf_analysis"

    def test_domain_keywords(self):
        result = self.router.route("What protein domains were annotated?")
        assert result == "domain_annotation"

    def test_no_match_returns_default(self):
        result = self.router.route("Hello there")
        assert result == "default"


class TestVectorStoreIntegration:
    """Integration tests for the vector store (require Ollama)."""

    @pytest.mark.integration
    @requires_network
    @patch('langchain_ollama.embeddings.OllamaEmbeddings.embed_documents') # 2. Intercept embedding requests
    @patch('langchain_ollama.embeddings.OllamaEmbeddings.embed_query')
    def test_build_and_query(self, mock_embed_query, mock_embed_docs, tmp_path):
        """Full build + query cycle. Requires Ollama running."""
        
        # 5. Define fake vector dimensions (768 numbers long, which matches nomic-embed-text)
        mock_embed_docs.return_value = [[0.1] * 768 for _ in range(20)]
        mock_embed_query.return_value = [0.1] * 768
        
        from knowledge_base.ingestion import TP53DocumentIngester
        from knowledge_base.vector_store import TP53VectorStore
        from config import settings
        settings.CHROMA_DIR = tmp_path / "chroma_test"
    
        ingester = TP53DocumentIngester()
        docs = ingester.load_curated_knowledge()
        chunks = ingester.chunk_documents(docs)
    
        store = TP53VectorStore()
        store.build(chunks[:20])

        results = store.similarity_search("R175H mutation cancer", k=3)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, _ in results)


class TestPipelineDataFormatting:
    """Test that pipeline data is correctly formatted for agents."""

    def test_format_mutations(self):
        from agents.rag_chain import TP53RAGChain
        chain = TP53RAGChain.__new__(TP53RAGChain)  # Skip __init__
        chain.llm = None
        chain.router = IntentRouter()
        chain.output_parser = None

        data = {
            "mutations": [
                {"position": 524, "amino_acid_change": "R175H"},
                {"position": 742, "amino_acid_change": "R248W"},
            ]
        }
        formatted = chain._format_pipeline_data(data)
        assert "R175H" in formatted
        assert "R248W" in formatted

    def test_format_empty_data(self):
        from agents.rag_chain import TP53RAGChain
        chain = TP53RAGChain.__new__(TP53RAGChain)
        chain.llm = None
        chain.router = IntentRouter()
        chain.output_parser = None

        formatted = chain._format_pipeline_data({})
        assert formatted == ""


class TestVizHelpers:
    """Visualization helpers in utils/viz.py — pure, LLM-free, edge-case safe."""

    def test_status_badge_states(self):
        from utils.viz import agent_status_badge
        assert "running" in agent_status_badge("mutation_analysis", "running")
        assert "complete" in agent_status_badge("drug_discovery", "complete", 1.4)
        assert "failed" in agent_status_badge("x", "failed", 0.2)

    def test_status_badge_unknown_state_non_empty(self):
        from utils.viz import agent_status_badge
        out = agent_status_badge("x", "not-a-state")
        assert out and "tp53-badge" in out  # degrades, never empty

    def test_vaf_timeline_valid(self):
        from utils.viz import animated_vaf_timeline
        fig = animated_vaf_timeline([0, 5, 10, 15], [50, 48, 45, 40])
        assert len(fig.frames) == 4
        assert fig.to_json()  # serializes for the browser

    def test_vaf_timeline_empty_and_mismatch(self):
        from utils.viz import animated_vaf_timeline
        assert animated_vaf_timeline([], []).to_json()
        assert animated_vaf_timeline([0, 5], [1]).to_json()  # length mismatch
        assert animated_vaf_timeline([0, 1], ["x", "y"]).to_json()  # non-numeric

    def test_vaf_marker_colour_logic(self):
        # rising VAF -> red marker, falling -> green
        from utils.viz import animated_vaf_timeline
        fig = animated_vaf_timeline([0, 1, 2], [10, 20, 15])
        colours = fig.frames[-1].data[0].marker.color
        assert colours[1] == "#ff4b4b" and colours[2] == "#2ecc71"

    def test_hotspot_bar_valid_and_bad_input(self):
        from utils.viz import animated_hotspot_bar
        fig = animated_hotspot_bar(["175", "248"], [8.0, 7.5])
        assert len(fig.frames) == 12 and fig.to_json()
        assert animated_hotspot_bar(["175"], ["nope"]).to_json()
        assert animated_hotspot_bar([], []).to_json()

    def test_architecture_frame_traces_uniform(self):
        # Plotly tweening requires identical trace count in every frame.
        from utils.viz import agent_architecture_diagram
        fig = agent_architecture_diagram(["a", "b", "c", "d", "e"])
        base = len(fig.data)
        assert base == 5
        assert all(len(f.data) == base for f in fig.frames)
        assert fig.to_json()

    def test_architecture_empty_and_spin_toggle(self):
        from utils.viz import agent_architecture_diagram
        assert agent_architecture_diagram([]).to_json()
        # spin disabled must still produce a valid figure
        assert agent_architecture_diagram(["a", "b"], spin_revolutions=0.0).to_json()

    def test_architecture_axes_zoom_locked(self):
        from utils.viz import agent_architecture_diagram
        fig = agent_architecture_diagram(["a", "b", "c"])
        assert fig.layout.xaxis.fixedrange is True
        assert fig.layout.yaxis.fixedrange is True

    def test_domain_legend_chart(self):
        from utils.viz import domain_legend_chart, P53_DOMAINS
        fig = domain_legend_chart()
        assert len(fig.data) == len(P53_DOMAINS)
        assert fig.to_json()

    def test_parse_residues_basic_and_dedup(self):
        from utils.viz import parse_residues
        assert parse_residues("175 248") == [175, 248, 273]
        assert parse_residues("273, 273, 100") == [100, 175, 248, 273]

    def test_parse_residues_empty_returns_hotspots(self):
        from utils.viz import parse_residues, CANONICAL_HOTSPOTS
        assert parse_residues("") == sorted(CANONICAL_HOTSPOTS)

    def test_parse_residues_out_of_range_dropped(self):
        from utils.viz import parse_residues
        # 99999 and 0 are out of 1..393 and must be dropped
        assert 99999 not in parse_residues("99999")
        assert 0 not in parse_residues("0")

    def test_viewer_html_injection_safe(self):
        from utils.viz import protein_viewer_html
        html = protein_viewer_html('2OCO"; alert(1)//', [175, 248, 273])
        assert "alert(1)" not in html          # script payload stripped
        assert "175,248,273" in html           # sanitized residues embedded
        assert "addLabel" in html              # hotspot labels present
        assert "s:1,e:66" in html              # domain ranges injected

    def test_viewer_html_default_pdb_on_garbage(self):
        from utils.viz import protein_viewer_html
        html = protein_viewer_html("!!!", [])
        assert "pdb:2OCJ" in html  # falls back to a safe default id

    def test_mutation_class(self):
        from utils.viz import mutation_class
        assert mutation_class("Y220C") == "y220c"
        assert mutation_class("R248W") == "contact"
        assert mutation_class("R175H") == "conformational"
        assert mutation_class("C176Y") == "zinc"
        assert mutation_class("P72R") == "other"

    def test_dock_candidates_ranking_shifts_by_mutation(self):
        from utils.viz import dock_candidates
        y = dock_candidates("Y220C")
        r = dock_candidates("R175H")
        assert y[0]["name"].startswith("PC14586")          # Y220C-specific wins for Y220C
        assert r[0]["name"].startswith("APR-246")          # reactivator wins for R175H
        assert [c["name"] for c in y] != [c["name"] for c in r]
        # ranks are 1..n, affinities sorted ascending (most negative first)
        assert [c["rank"] for c in y] == list(range(1, len(y) + 1))
        assert all(y[i]["affinity"] <= y[i + 1]["affinity"] for i in range(len(y) - 1))

    def test_dock_mdm2_penalised_on_lof(self):
        from utils.viz import dock_candidates
        r = dock_candidates("R175H")
        idasa = [c for c in r if "Idasanutlin" in c["name"]][0]
        assert idasa["rank"] == len(r)  # MDM2 inhibitor ranks last on a LOF mutant

    def test_docking_affinity_chart_valid_and_empty(self):
        from utils.viz import dock_candidates, docking_affinity_chart
        assert docking_affinity_chart(dock_candidates("R248W")).to_json()
        assert docking_affinity_chart([]).to_json()  # never empty

    def test_docking_pose_html_injection_safe(self):
        from utils.viz import docking_pose_html
        html = docking_pose_html('2OCJ"; alert(1)//', [175, 248], "Evil<script>", -7.5)
        assert "alert(1)" not in html
        assert "175,248" in html and "kcal/mol" in html

    def test_tnm_stage_bar_all_groups(self):
        from utils.viz import tnm_stage_bar
        for s in ["I", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IV", "", "?"]:
            assert tnm_stage_bar(s).to_json()  # never empty, valid for any input

    def test_pathogenicity_gauge(self):
        from utils.viz import pathogenicity_gauge
        for s in ["pathogenic", "likely_benign", "vus", "benign", "", "weird"]:
            assert pathogenicity_gauge(s, 0.9).to_json()
        assert pathogenicity_gauge("pathogenic", None).to_json()

    def test_tme_donut(self):
        from utils.viz import tme_donut
        assert tme_donut(0.35, "immune-hot").to_json()
        assert tme_donut("bad-input", "").to_json()   # non-numeric tolerated
        assert tme_donut(2.0, "cold").to_json()        # clamped

    def test_vaf_gauge(self):
        from utils.viz import vaf_gauge
        assert vaf_gauge(47.3).to_json()
        assert vaf_gauge("x").to_json()                # non-numeric tolerated

    def test_pathway_diverging_bar(self):
        from utils.viz import pathway_diverging_bar
        assert pathway_diverging_bar(["MDM2", "BAX"], ["CDKN1A"]).to_json()
        assert pathway_diverging_bar([], []).to_json()  # never empty

    def test_african_atlas_map(self):
        from utils.viz import african_atlas_map
        assert african_atlas_map({"Kenya": 86, "Nigeria": 88}).to_json()
        assert african_atlas_map({}).to_json()  # never empty

    def test_african_burden_bar(self):
        from utils.viz import african_burden_bar
        rows = [{"title": "A", "burden_score": 90}, {"title": "B", "burden_score": 70}]
        assert african_burden_bar(rows).to_json()
        assert african_burden_bar([]).to_json()  # never empty


class TestAfricanTP53Atlas:
    """Agent #17 — African TP53 Atlas (curated regional epidemiology)."""

    def _agent(self):
        from agents.african_atlas import AfricanTP53Atlas
        return AfricanTP53Atlas()

    def test_mutation_lookup_r249s(self):
        out = self._agent().profile(mutation="R249S")
        assert out["status"] == "success"
        ids = [p["id"] for p in out["atlas"]["matched_profiles"]]
        assert "aflatoxin_hcc" in ids

    def test_codon_only_match(self):
        # "249" alone should still match the R249S profile
        out = self._agent().profile(mutation="249")
        ids = [p["id"] for p in out["atlas"]["matched_profiles"]]
        assert "aflatoxin_hcc" in ids

    def test_region_lookup_kenya(self):
        out = self._agent().profile(region="Kenya")
        assert len(out["atlas"]["matched_profiles"]) >= 2
        assert not out["broadened"]

    def test_cancer_lookup_liver(self):
        out = self._agent().profile(cancer_type="liver")
        ids = [p["id"] for p in out["atlas"]["matched_profiles"]]
        assert "aflatoxin_hcc" in ids

    def test_unknown_falls_back_to_continental(self):
        out = self._agent().profile(mutation="ZZZ999")
        assert out["broadened"] is True
        assert len(out["atlas"]["matched_profiles"]) >= 4  # full atlas, never empty

    def test_no_filters_returns_overview(self):
        out = self._agent().profile()
        assert out["atlas"]["query"] == "continental overview"
        assert len(out["atlas"]["countries"]) > 5

    def test_cervical_is_wild_type(self):
        # HPV-driven cervical cancer: TP53 typically wild-type (E6-inactivated)
        out = self._agent().profile(cancer_type="cervical")
        cerv = [p for p in out["atlas"]["matched_profiles"] if p["id"] == "cervical_hpv"][0]
        assert cerv["key_mutations"] == []

    def test_output_is_json_serializable(self):
        import json
        out = self._agent().profile(mutation="R249S")
        assert json.dumps(out)  # no non-serialisable objects

    def test_disclaimer_and_sources_present(self):
        out = self._agent().profile(region="West Africa")
        assert out["atlas"]["disclaimer"]
        assert all(p["sources"] for p in out["atlas"]["matched_profiles"])

    def test_country_burden_map(self):
        cb = self._agent().country_burden()
        assert "Kenya" in cb and 0 <= cb["Kenya"] <= 100

    def test_convenience_function(self):
        from agents.african_atlas import atlas_profile
        assert atlas_profile(mutation="R249S")["status"] == "success"

    def test_registered_in_agent_registry(self):
        from config.settings import AGENT_REGISTRY
        assert "african_tp53_atlas" in AGENT_REGISTRY
        assert AGENT_REGISTRY["african_tp53_atlas"]["keywords"]


class TestClinVarConflictChecker:
    """Hallucination guard — AI classifications vs ClinVar."""

    def _agent(self):
        from agents.clinvar_conflict_checker import ClinVarConflictChecker
        return ClinVarConflictChecker()

    def test_extract_mutations(self):
        from agents.clinvar_conflict_checker import extract_mutations
        muts = extract_mutations("p.R175H and R248W; benign P72R; ignore A9999Z")
        assert "R175H" in muts and "R248W" in muts and "P72R" in muts
        assert all(not m.endswith("9999Z") for m in muts)  # codon range enforced

    def test_high_conflict_benign_vs_pathogenic(self):
        out = self._agent().check(mutation="R175H", ai_classification="benign")
        f = out["findings"][0]
        assert f["conflict"] is True and f["severity"] == "high"
        assert out["verdict"] == "conflict_high"

    def test_concordant(self):
        out = self._agent().check(mutation="R175H", ai_classification="pathogenic")
        assert out["verdict"] == "concordant"
        assert out["findings"][0]["conflict"] is False

    def test_medium_conflict_vus(self):
        f = self._agent().check(mutation="R248W", ai_classification="VUS")["findings"][0]
        assert f["severity"] == "medium" and f["conflict"] is True

    def test_unknown_mutation_not_in_reference(self):
        f = self._agent().check(mutation="A159V", ai_classification="benign")["findings"][0]
        assert f["severity"] == "unknown" and f["conflict"] is False

    def test_no_claim_no_conflict(self):
        # mutation present but no classification claimed -> not a conflict
        f = self._agent().check(mutation="R175H")["findings"][0]
        assert f["conflict"] is False and f["severity"] == "none"

    def test_free_text_conflict_detection(self):
        out = self._agent().check(text="R175H is a benign polymorphism.")
        assert out["conflicts_found"] == 1
        assert out["findings"][0]["severity"] == "high"

    def test_free_text_concordant(self):
        out = self._agent().check(text="R175H is pathogenic in this tumour.")
        assert out["verdict"] == "concordant"

    def test_never_empty_no_mutation(self):
        out = self._agent().check(text="general question about cancer")
        assert out["verdict"] == "no_claims" and out["findings"] == []

    def test_codon_only_clinvar_lookup(self):
        # a different alt at codon 175 should still map to the codon's ClinVar call
        assert self._agent().clinvar_classification("R175C") == "pathogenic"

    def test_evidence_url_present(self):
        f = self._agent().check(mutation="R249S", ai_classification="benign")["findings"][0]
        assert "clinvar" in f["evidence_url"].lower() and "R249S" in f["evidence_url"]

    def test_output_json_serialisable(self):
        import json
        assert json.dumps(self._agent().check(mutation="R175H", ai_classification="benign"))

    def test_convenience_function_and_registry(self):
        from agents.clinvar_conflict_checker import check_conflicts
        from config.settings import AGENT_REGISTRY
        assert check_conflicts(mutation="R175H", ai_classification="benign")["status"] == "success"
        assert "clinvar_conflict_checker" in AGENT_REGISTRY

    def test_conflict_chart_viz(self):
        from utils.viz import clinvar_conflict_chart
        f = self._agent().check(text="R175H benign; R248W pathogenic")["findings"]
        assert clinvar_conflict_chart(f).to_json()
        assert clinvar_conflict_chart([]).to_json()  # never empty


class TestChEMBLClient:
    """TP53-pathway drug data — offline-first with live ChEMBL augmentation."""

    _MOCK = {"mechanisms": [
        {"molecule_chembl_id": "CHEMBL999", "mechanism_of_action": "MDM2 inhibitor",
         "max_phase": 2},
        {"molecule_chembl_id": "CHEMBL998", "mechanism_of_action": "p53 stabiliser",
         "max_phase": None},
        {"bad": "row"},
        {"molecule_chembl_id": None},
    ]}

    def test_offline_curated_never_empty(self):
        from utils.chembl_client import ChEMBLClient
        out = ChEMBLClient().compounds(use_live=False)
        assert out["count"] >= 5 and out["live"] is False
        assert out["source"] == "curated"

    def test_curated_sorted_by_phase(self):
        from utils.chembl_client import ChEMBLClient
        comp = ChEMBLClient().compounds(use_live=False)["compounds"]
        phases = [c.get("max_phase") or -1 for c in comp]
        assert phases == sorted(phases, reverse=True)

    def test_phase_label_and_url_present(self):
        from utils.chembl_client import ChEMBLClient
        for c in ChEMBLClient().compounds(use_live=False)["compounds"]:
            assert "phase_label" in c and "chembl_url" in c

    def test_parse_mechanisms_pure(self):
        from utils.chembl_client import parse_mechanisms
        recs = parse_mechanisms(self._MOCK, "MDM2")
        assert len(recs) == 2  # 2 valid rows, bad/None dropped
        assert recs[0]["chembl_id"] == "CHEMBL999" and recs[0]["max_phase"] == 2

    def test_parse_mechanisms_none_safe(self):
        from utils.chembl_client import parse_mechanisms
        assert parse_mechanisms(None, "MDM2") == []
        assert parse_mechanisms({"x": 1}, "MDM2") == []

    def test_live_merge_when_network_ok(self):
        from unittest.mock import patch
        from utils.chembl_client import ChEMBLClient
        c = ChEMBLClient()
        with patch.object(ChEMBLClient, "_get_json", return_value=self._MOCK):
            out = c.compounds(use_live=True)
        assert out["live"] is True and out["source"] == "chembl-live+curated"
        assert any(d.get("source") == "chembl-live" for d in out["compounds"])

    def test_graceful_when_network_fails(self):
        from unittest.mock import patch
        from utils.chembl_client import ChEMBLClient
        c = ChEMBLClient()
        with patch.object(ChEMBLClient, "_get_json", return_value=None):
            out = c.compounds(use_live=True)
        assert out["live"] is False and out["count"] >= 5  # fell back to curated

    def test_caching(self):
        from unittest.mock import patch
        from utils.chembl_client import ChEMBLClient
        c = ChEMBLClient()
        with patch.object(ChEMBLClient, "_get_json", return_value=self._MOCK) as m:
            c.fetch_target_drugs("MDM2")
            c.fetch_target_drugs("MDM2")  # second call should hit cache
        assert m.call_count == 1

    def test_unknown_target_returns_empty(self):
        from utils.chembl_client import ChEMBLClient
        assert ChEMBLClient().fetch_target_drugs("NOPE") == []

    def test_convenience_function(self):
        from utils.chembl_client import tp53_pathway_drugs
        assert tp53_pathway_drugs(use_live=False)["count"] >= 5

    def test_phase_chart_viz(self):
        from utils.viz import chembl_phase_chart
        from utils.chembl_client import ChEMBLClient
        comp = ChEMBLClient().compounds(use_live=False)["compounds"]
        assert chembl_phase_chart(comp).to_json()
        assert chembl_phase_chart([]).to_json()  # never empty


class TestBenchmarkScoring:
    """Pure scoring helpers in benchmarks/scoring.py."""

    def test_normalize_synonyms(self):
        from benchmarks.scoring import normalize_significance as nz
        assert nz("Likely Pathogenic") == "likely_pathogenic"
        assert nz("Uncertain significance") == "vus"
        assert nz("NEUTRAL") == "benign"
        assert nz(None) == ""

    def test_bucket_collapse(self):
        from benchmarks.scoring import significance_bucket as sb
        assert sb("pathogenic") == "pathogenic_leaning"
        assert sb("likely_pathogenic") == "pathogenic_leaning"
        assert sb("benign") == "benign_leaning"
        assert sb("vus") == "uncertain"
        assert sb("garbage") == "uncertain"

    def test_score_variant_exact_and_iarc(self):
        from benchmarks.scoring import score_variant
        r = score_variant(
            {"clinical_significance": "pathogenic", "iarc_classification": "R1"},
            {"mutation": "R175H", "expected_significance": "pathogenic", "expected_iarc": "R1"},
        )
        assert r["exact_match"] and r["concordant"] and r["iarc_match"] is True

    def test_score_variant_concordant_not_exact(self):
        from benchmarks.scoring import score_variant
        r = score_variant(
            {"clinical_significance": "likely_pathogenic"},
            {"mutation": "X", "expected_significance": "pathogenic", "expected_iarc": None},
        )
        assert r["exact_match"] is False
        assert r["concordant"] is True
        assert r["iarc_match"] is None  # not applicable

    def test_score_variant_missing_prediction(self):
        from benchmarks.scoring import score_variant
        r = score_variant({}, {"mutation": "Y", "expected_significance": "benign",
                               "expected_iarc": None})
        assert r["exact_match"] is False and r["predicted_significance"] == "(none)"

    def test_aggregate_empty(self):
        from benchmarks.scoring import aggregate
        m = aggregate([])
        assert m["n"] == 0 and m["precision"] == 0.0 and "note" in m

    def test_aggregate_metrics_and_confusion(self):
        from benchmarks.scoring import score_variant, aggregate
        rs = [
            score_variant({"clinical_significance": "pathogenic"},
                          {"mutation": "A", "expected_significance": "pathogenic"}),
            score_variant({"clinical_significance": "vus"},
                          {"mutation": "B", "expected_significance": "benign"}),
        ]
        m = aggregate(rs)
        assert m["n"] == 2 and m["tp"] == 1 and m["fp"] == 0
        assert 0.0 <= m["precision"] <= 1.0


class TestBenchmarkRunner:
    """Runner orchestration in benchmarks/run_benchmark.py (no LLM needed)."""

    def test_load_ground_truth_real_file(self):
        from benchmarks.run_benchmark import load_ground_truth
        gt = load_ground_truth()
        assert len(gt["variants"]) >= 5
        assert all("mutation" in v for v in gt["variants"])

    def test_load_ground_truth_missing_file_graceful(self):
        from pathlib import Path
        from benchmarks.run_benchmark import load_ground_truth
        gt = load_ground_truth(Path("does_not_exist_xyz.json"))
        assert gt["variants"] == []  # degrades, no raise

    def test_run_benchmark_with_oracle(self):
        from benchmarks.run_benchmark import run_benchmark, load_ground_truth
        gt = load_ground_truth()

        def oracle(mut):
            for v in gt["variants"]:
                if v["mutation"] == mut:
                    return {"clinical_significance": v["expected_significance"],
                            "iarc_classification": v["expected_iarc"]}
            return {}

        rep = run_benchmark(classifier=oracle, ground_truth=gt)
        assert rep["metrics"]["exact_accuracy"] == 1.0
        assert rep["metrics"]["recall"] == 1.0

    def test_run_benchmark_classifier_exception_is_caught(self):
        from benchmarks.run_benchmark import run_benchmark
        def boom(mut):
            raise RuntimeError("simulated agent failure")
        rep = run_benchmark(classifier=boom,
                            ground_truth={"variants": [{"mutation": "R175H",
                                          "expected_significance": "pathogenic"}]})
        assert rep["metrics"]["n"] == 1  # ran without crashing

    def test_render_markdown_non_empty(self):
        from benchmarks.run_benchmark import run_benchmark, render_markdown
        rep = run_benchmark(classifier=lambda m: {}, ground_truth={"variants": []})
        md = render_markdown(rep)
        assert md and "Benchmark Report" in md


class TestVariantCuratorClassification:
    """Regression lock for the hotspot-key bug the benchmark surfaced:
    multi-digit codons must classify correctly, not fall through to VUS."""

    def test_known_hotspots_are_pathogenic(self):
        from agents.variant_curator import VariantCurator
        c = VariantCurator()
        for mut in ["R175H", "R248W", "R273H", "Y220C", "R282W", "R249S"]:
            out = c.classify(mut)["classification"]
            assert out["clinical_significance"] == "pathogenic", mut

    def test_hotspot_function_class_is_lof(self):
        from agents.variant_curator import VariantCurator
        out = VariantCurator().classify("R175H")["classification"]
        assert out["function_class"] == "loss-of-function"

    def test_iarc_classification_resolved(self):
        from agents.variant_curator import VariantCurator
        out = VariantCurator().classify("R175H")["classification"]
        assert out["iarc_classification"] == "R1"

    def test_non_hotspot_is_vus_not_pathogenic(self):
        from agents.variant_curator import VariantCurator
        out = VariantCurator().classify("A159V")["classification"]
        assert out["clinical_significance"] == "vus"

    def test_benign_control_not_called_pathogenic(self):
        # P72R is a benign polymorphism — must never be reported pathogenic.
        from agents.variant_curator import VariantCurator
        out = VariantCurator().classify("P72R")["classification"]
        assert out["clinical_significance"] != "pathogenic"

    def test_codon_extracted_correctly(self):
        from agents.variant_curator import VariantCurator
        out = VariantCurator().classify("R273H")["classification"]
        assert out["codon"] == 273
