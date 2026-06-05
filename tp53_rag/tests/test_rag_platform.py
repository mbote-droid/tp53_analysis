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


class TestClinicalTrialsMatcher:
    """ClinicalTrials.gov matcher — offline-first, Kenya/Africa prioritised."""

    # mock v2 payload: a US Phase-2 and a Kenya Phase-3 recruiting study
    _PAYLOAD = {"studies": [
        {"protocolSection": {
            "identificationModule": {"nctId": "NCT001", "briefTitle": "US study"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "designModule": {"phases": ["PHASE2"]},
            "conditionsModule": {"conditions": ["Breast Cancer"]},
            "contactsLocationsModule": {"locations": [{"country": "United States"}]}}},
        {"protocolSection": {
            "identificationModule": {"nctId": "NCT002", "briefTitle": "Kenya study"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "designModule": {"phases": ["PHASE3"]},
            "conditionsModule": {"conditions": ["Liver Cancer"]},
            "contactsLocationsModule": {"locations": [{"country": "Kenya"},
                                                      {"country": "Nigeria"}]}}},
    ]}

    def test_parse_studies(self):
        from agents.clinical_trials import parse_studies
        recs = parse_studies(self._PAYLOAD)
        assert len(recs) == 2
        k = [r for r in recs if r["nct_id"] == "NCT002"][0]
        assert k["african_priority"] and k["kenya_site"] and k["phase"] == "Phase 3"

    def test_parse_none_safe(self):
        from agents.clinical_trials import parse_studies
        assert parse_studies(None) == [] and parse_studies({"x": 1}) == []

    def test_phase_normalisation(self):
        from agents.clinical_trials import _norm_phase
        assert _norm_phase(["PHASE1", "PHASE2"]) == "Phase 1/2"
        assert _norm_phase(["NA"]) == "N/A" and _norm_phase(None) == "N/A"

    def test_kenya_sorts_first(self):
        from unittest.mock import patch
        from agents.clinical_trials import ClinicalTrialsMatcher
        with patch.object(ClinicalTrialsMatcher, "_get_json", return_value=self._PAYLOAD):
            out = ClinicalTrialsMatcher().search(mutation="R249S", cancer_type="liver")
        assert out["trials"][0]["nct_id"] == "NCT002"  # Kenya first
        assert out["kenya_count"] == 1 and out["african_count"] == 1

    def test_graceful_fallback_never_empty(self):
        from unittest.mock import patch
        from agents.clinical_trials import ClinicalTrialsMatcher
        with patch.object(ClinicalTrialsMatcher, "_get_json", return_value=None):
            out = ClinicalTrialsMatcher().search(mutation="R175H", cancer_type="breast")
        assert out["live"] is False and out["count"] >= 1
        assert out["trials"][0]["source"] == "curated-search"

    def test_offline_mode(self):
        from agents.clinical_trials import ClinicalTrialsMatcher
        out = ClinicalTrialsMatcher().search(use_live=False)
        assert out["count"] >= 1 and out["live"] is False

    def test_phase_filter_does_not_empty(self):
        # a payload with only a Phase-1 study should still return something
        from unittest.mock import patch
        from agents.clinical_trials import ClinicalTrialsMatcher
        p1 = {"studies": [{"protocolSection": {
            "identificationModule": {"nctId": "NCT9", "briefTitle": "early"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "designModule": {"phases": ["PHASE1"]},
            "contactsLocationsModule": {"locations": [{"country": "Kenya"}]}}}]}
        with patch.object(ClinicalTrialsMatcher, "_get_json", return_value=p1):
            out = ClinicalTrialsMatcher().search(mutation="X", cancer_type="y")
        assert out["count"] >= 1  # phase filter kept it rather than going empty

    def test_caching(self):
        from unittest.mock import patch
        from agents.clinical_trials import ClinicalTrialsMatcher
        m = ClinicalTrialsMatcher()
        with patch.object(ClinicalTrialsMatcher, "_get_json",
                          return_value=self._PAYLOAD) as mock:
            m.search(mutation="R175H", cancer_type="breast")
            m.search(mutation="R175H", cancer_type="breast")
        assert mock.call_count == 1  # second hit cache

    def test_output_json_serialisable(self):
        import json
        from unittest.mock import patch
        from agents.clinical_trials import ClinicalTrialsMatcher
        with patch.object(ClinicalTrialsMatcher, "_get_json", return_value=self._PAYLOAD):
            out = ClinicalTrialsMatcher().search(mutation="R175H", cancer_type="breast")
        assert json.dumps(out)

    def test_convenience_and_registry(self):
        from agents.clinical_trials import match_trials
        from config.settings import AGENT_REGISTRY
        assert match_trials(use_live=False)["status"] == "success"
        assert "clinical_trials_matcher" in AGENT_REGISTRY

    def test_trials_viz(self):
        from utils.viz import trials_priority_chart
        from agents.clinical_trials import parse_studies
        assert trials_priority_chart(parse_studies(self._PAYLOAD)).to_json()
        assert trials_priority_chart([]).to_json()  # never empty


class TestVCFParser:
    """VCF parsing — TP53 filter + honest HGVS-based amino-acid extraction."""

    def test_sample_vcf_counts(self):
        from utils.vcf_parser import parse_vcf_text, sample_vcf
        out = parse_vcf_text(sample_vcf())
        assert out["tp53_count"] == 3  # 2 annotated hotspots + 1 unannotated
        assert out["skipped"] == 0

    def test_non_tp53_filtered_out(self):
        from utils.vcf_parser import parse_vcf_text, sample_vcf
        chroms = {v["chrom"] for v in parse_vcf_text(sample_vcf())["variants"]}
        assert chroms == {"17"}  # the chr1 line is excluded

    def test_protein_change_3letter(self):
        from utils.vcf_parser import extract_protein_change
        assert extract_protein_change("HGVS.p=p.Arg175His") == "R175H"
        assert extract_protein_change("p.Arg196Ter") == "R196*"

    def test_protein_change_1letter(self):
        from utils.vcf_parser import extract_protein_change
        assert extract_protein_change("p.R273H") == "R273H"

    def test_protein_change_none(self):
        from utils.vcf_parser import extract_protein_change
        assert extract_protein_change("no annotation") is None
        assert extract_protein_change("") is None

    def test_hotspot_flagging(self):
        from utils.vcf_parser import parse_vcf_text, sample_vcf
        hot = [v for v in parse_vcf_text(sample_vcf())["variants"] if v["is_hotspot"]]
        assert {v["amino_acid_change"] for v in hot} == {"R175H", "R248W"}

    def test_unannotated_is_genomic_only(self):
        from utils.vcf_parser import parse_vcf_text, sample_vcf
        un = [v for v in parse_vcf_text(sample_vcf())["variants"] if not v["annotated"]]
        assert len(un) == 1 and un[0]["amino_acid_change"] is None

    def test_locus_detection_both_builds(self):
        from utils.vcf_parser import is_tp53_locus
        assert is_tp53_locus("chr17", 7675088)   # GRCh38
        assert is_tp53_locus("17", 7578406)       # hg19
        assert not is_tp53_locus("1", 12345)
        assert not is_tp53_locus("17", 999999999)

    def test_malformed_lines_skipped(self):
        from utils.vcf_parser import parse_vcf_text
        out = parse_vcf_text("17\tNOTANUMBER\t.\tC\tT\n17\t7675088\t.\tC\tT\t.\tPASS\tp.Arg175His")
        assert out["skipped"] == 1 and out["tp53_count"] == 1

    def test_empty_input(self):
        from utils.vcf_parser import parse_vcf_text
        out = parse_vcf_text("")
        assert out["tp53_count"] == 0 and out["variants"] == []

    def test_bytes_parsing(self):
        from utils.vcf_parser import parse_vcf_bytes, sample_vcf
        assert parse_vcf_bytes(sample_vcf().encode())["tp53_count"] == 3

    def test_pipeline_data_compatible(self):
        from utils.vcf_parser import parse_vcf_text, sample_vcf
        v = parse_vcf_text(sample_vcf())["variants"][0]
        for k in ("gene", "chrom", "pos", "ref", "alt", "amino_acid_change"):
            assert k in v
        assert v["gene"] == "TP53"

    def test_vcf_viz(self):
        from utils.viz import vcf_variant_chart
        from utils.vcf_parser import parse_vcf_text, sample_vcf
        vs = parse_vcf_text(sample_vcf())["variants"]
        assert vcf_variant_chart(vs).to_json()
        assert vcf_variant_chart([]).to_json()  # never empty


class TestINDGenerator:
    """IND draft generator — rule-based regulatory scaffold."""

    def _gen(self):
        from agents.ind_generator import INDGenerator
        return INDGenerator()

    def test_generates_six_sections(self):
        r = self._gen().generate("R175H", "breast cancer",
                                  [{"name": "APR-246", "mechanism": "Mutant p53 reactivator"}])
        assert r["status"] == "success" and r["section_count"] == 6
        assert r["draft"]["lead_candidate"] == "APR-246"

    def test_strategy_detection(self):
        from agents.ind_generator import _strategy_for
        assert _strategy_for("MDM2 inhibitor") == "mdm2_inhibitor"
        assert _strategy_for("Y220C pocket stabiliser") == "stabiliser"
        assert _strategy_for("Mutant p53 reactivator") == "reactivator"
        assert _strategy_for("carboplatin DNA crosslink") == "chemotherapy"

    def test_never_empty_without_candidates(self):
        r = self._gen().generate("R248W", "")
        assert r["section_count"] == 6
        assert r["draft"]["lead_candidate"] == "TP53-targeting candidate"

    def test_empty_mutation_handled(self):
        r = self._gen().generate("", "")
        assert r["status"] == "success" and r["section_count"] == 6

    def test_readiness_pct(self):
        r = self._gen().generate("R175H", "lung")
        assert 0 <= r["readiness_pct"] <= 100

    def test_sections_have_required_fields(self):
        r = self._gen().generate("R273H", "ovarian")
        for s in r["draft"]["sections"]:
            assert s["number"] and s["title"] and s["content"]

    def test_mutation_appears_in_draft(self):
        r = self._gen().generate("Y220C", "sarcoma")
        joined = " ".join(s["content"] for s in r["draft"]["sections"])
        assert "Y220C" in joined

    def test_render_markdown(self):
        g = self._gen()
        md = g.render_markdown(g.generate("R175H", "breast", [{"name": "X", "mechanism": "MDM2 inhibitor"}]))
        assert "# DRAFT IND" in md and md.count("## ") == 6
        assert "DRAFT scaffold" in md

    def test_render_markdown_empty_safe(self):
        assert self._gen().render_markdown({}) and self._gen().render_markdown(None)

    def test_render_markdown_malformed_draft(self):
        # regression: a non-dict draft / non-dict sections must not crash
        g = self._gen()
        assert g.render_markdown({"draft": "notadict"})
        assert g.render_markdown({"draft": {"sections": ["x", None, {"number": "1"}]}})

    def test_disclaimer_present(self):
        r = self._gen().generate("R175H", "breast")
        assert "not a" in r["draft"]["disclaimer"].lower()

    def test_json_serialisable_and_registry(self):
        import json
        from config.settings import AGENT_REGISTRY
        from agents.ind_generator import generate_ind
        assert json.dumps(generate_ind("R175H", "breast"))
        assert "ind_generator" in AGENT_REGISTRY

    def test_ind_section_viz(self):
        from utils.viz import ind_section_chart
        r = self._gen().generate("R175H", "breast")
        assert ind_section_chart(r).to_json()
        assert ind_section_chart({}).to_json()  # never empty


class TestSyntheticLethality:
    """Synthetic-lethal target modeler (DepMap-derived, curated)."""

    def _m(self):
        from agents.synthetic_lethality import SyntheticLethalityModeler
        return SyntheticLethalityModeler()

    def test_returns_ranked_targets(self):
        r = self._m().model("R175H")
        assert r["status"] == "success" and r["count"] >= 5
        scores = [t["sl_score"] for t in r["targets"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_target_is_high_evidence(self):
        r = self._m().model("R248W")
        assert r["targets"][0]["evidence"] == "high"
        assert r["top_target"] in {"WEE1", "ATR", "CHEK1"}

    def test_evidence_filter(self):
        r = self._m().model("R175H", min_evidence="high")
        assert all(t["evidence"] == "high" for t in r["targets"])
        assert r["count"] >= 1

    def test_never_empty(self):
        assert self._m().model("")["count"] >= 5
        assert self._m().model("nonsense")["count"] >= 5

    def test_network_edges(self):
        r = self._m().model("R273H")
        assert len(r["network_edges"]) == r["count"]
        assert all(e["source"] == "TP53" for e in r["network_edges"])

    def test_known_targets_present(self):
        genes = {t["gene"] for t in self._m().model("R175H")["targets"]}
        assert {"WEE1", "ATR"}.issubset(genes)

    def test_targets_have_fields(self):
        for t in self._m().model("R175H")["targets"]:
            for k in ("gene", "mechanism", "evidence", "druggability", "source"):
                assert t.get(k) is not None

    def test_disclaimer(self):
        assert "research" in self._m().model("R175H")["disclaimer"].lower()

    def test_json_and_registry(self):
        import json
        from config.settings import AGENT_REGISTRY
        from agents.synthetic_lethality import model_synthetic_lethality
        assert json.dumps(model_synthetic_lethality("R175H"))
        assert "synthetic_lethality" in AGENT_REGISTRY

    def test_network_viz(self):
        from utils.viz import synthetic_lethal_network
        r = self._m().model("R175H")
        assert synthetic_lethal_network(r).to_json()
        assert synthetic_lethal_network({}).to_json()  # never empty


class TestPubMedCitations:
    """PubMed inline citations — Entrez E-utilities, offline-honest fallback."""

    _ES = {"esearchresult": {"idlist": ["111", "222"]}}
    _SUMM = {"result": {"uids": ["111", "222"],
             "111": {"title": "p53 reactivation.", "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                     "pubdate": "2020 Jan", "source": "Nature"},
             "222": {"title": "WEE1 inhibition.", "authors": [{"name": "Lee K"}],
                     "pubdate": "2019", "source": "Cell"}}}

    def test_parse_esearch(self):
        from utils.pubmed_citations import parse_esearch
        assert parse_esearch(self._ES) == ["111", "222"]
        assert parse_esearch(None) == [] and parse_esearch({"x": 1}) == []

    def test_parse_esummary(self):
        from utils.pubmed_citations import parse_esummary
        recs = parse_esummary(self._SUMM)
        assert len(recs) == 2 and recs[0]["pmid"] == "111" and recs[0]["year"] == "2020"
        assert recs[0]["authors"] == ["Smith J", "Doe A"]

    def test_parse_esummary_none_safe(self):
        from utils.pubmed_citations import parse_esummary
        assert parse_esummary(None) == [] and parse_esummary({}) == []

    def test_format_citation(self):
        from utils.pubmed_citations import format_citation
        s = format_citation({"pmid": "111", "authors": ["Smith J", "Doe A"],
                             "source": "Nature", "year": "2020"})
        assert "Smith J et al." in s and "[PMID: 111]" in s

    def test_live_cite(self):
        from unittest.mock import patch
        from utils.pubmed_citations import PubMedClient
        c = PubMedClient()
        with patch.object(PubMedClient, "_get_json", side_effect=[self._ES, self._SUMM]):
            out = c.cite("R175H")
        assert out["live"] and out["count"] == 2
        assert out["inline"][0].startswith("Smith J et al.")

    def test_fallback_no_fake_pmids(self):
        from unittest.mock import patch
        from utils.pubmed_citations import PubMedClient
        c = PubMedClient()
        with patch.object(PubMedClient, "_get_json", return_value=None):
            out = c.cite("R248W")
        assert out["live"] is False and out["count"] == 1
        assert out["citations"][0]["pmid"] is None
        assert "pubmed" in out["citations"][0]["url"].lower()

    def test_tp53_prepended(self):
        from unittest.mock import patch
        from utils.pubmed_citations import PubMedClient
        c = PubMedClient()
        with patch.object(PubMedClient, "_get_json", return_value=None):
            out = c.cite("R175H")
        assert out["query"].startswith("TP53")

    def test_caching(self):
        from unittest.mock import patch
        from utils.pubmed_citations import PubMedClient
        c = PubMedClient()
        with patch.object(PubMedClient, "_get_json", side_effect=[self._ES, self._SUMM]) as m:
            c.cite("R175H")
            c.cite("R175H")  # second hit cache
        assert m.call_count == 2  # esearch + esummary once, not twice each

    def test_empty_query_handled(self):
        from unittest.mock import patch
        from utils.pubmed_citations import PubMedClient
        with patch.object(PubMedClient, "_get_json", return_value=None):
            out = PubMedClient().cite("")
        assert out["status"] == "success" and out["count"] >= 1

    def test_search_url_helpers(self):
        from utils.pubmed_citations import pubmed_search_url, pubmed_abstract_url
        assert "term=TP53" in pubmed_search_url("TP53 R175H")
        assert pubmed_abstract_url("123").endswith("/123/")

    def test_convenience_function(self):
        from unittest.mock import patch
        from utils.pubmed_citations import pubmed_cite, PubMedClient
        with patch.object(PubMedClient, "_get_json", return_value=None):
            assert pubmed_cite("R175H")["status"] == "success"

    def test_cite_survives_fetch_exception(self):
        # regression: an unexpected error in the fetch layer must degrade,
        # not crash
        from unittest.mock import patch
        from utils.pubmed_citations import PubMedClient
        with patch.object(PubMedClient, "_get_json", side_effect=Exception("boom")):
            out = PubMedClient().cite("R175H")
        assert out["status"] == "success" and out["live"] is False and out["count"] >= 1


class TestMolecularDocking:
    """Molecular docking agent — AutoDock Vina or honest heuristic estimate."""

    def _a(self):
        from agents.molecular_docking import MolecularDockingAgent
        return MolecularDockingAgent()

    def test_parse_vina_output_best_mode(self):
        from agents.molecular_docking import parse_vina_output
        txt = ("mode | affinity\n----+----\n   1       -8.4   0.0  0.0\n"
               "   2       -7.9   1.2  2.3\n   3       -6.1   3.0  4.0\n")
        assert parse_vina_output(txt) == -8.4

    def test_parse_vina_output_none_safe(self):
        from agents.molecular_docking import parse_vina_output
        assert parse_vina_output("no table") is None
        assert parse_vina_output("") is None and parse_vina_output(None) is None

    def test_strategy_detection(self):
        from agents.molecular_docking import _strategy
        assert _strategy("MDM2 inhibitor") == "mdm2_inhibitor"
        assert _strategy("Y220C pocket stabiliser") == "stabiliser"
        assert _strategy("p53 reactivator") == "reactivator"
        assert _strategy("carboplatin DNA crosslink") == "chemotherapy"

    def test_fallback_is_heuristic_when_no_vina(self):
        # no PDBQT inputs => always the heuristic estimate, clearly labelled
        r = self._a().dock("R175H", "APR-246", mechanism="Mutant p53 reactivator")
        assert r["method"] == "heuristic_estimate"
        assert r["binding_affinity"] is not None and r["status"] == "success"

    def test_uses_vina_when_available_and_parses(self):
        # simulate Vina present + a successful run
        from unittest.mock import patch
        import agents.molecular_docking as md

        class _P:
            stdout = "mode | affinity\n   1   -9.9  0 0\n"
        with patch.object(md, "vina_available", return_value=True), \
             patch.object(md.subprocess, "run", return_value=_P()):
            r = self._a().dock("R175H", "X", receptor_pdbqt="r.pdbqt",
                               ligand_pdbqt="l.pdbqt")
        assert r["method"] == "autodock_vina" and r["binding_affinity"] == -9.9

    def test_vina_run_failure_falls_back(self):
        from unittest.mock import patch
        import agents.molecular_docking as md
        with patch.object(md, "vina_available", return_value=True), \
             patch.object(md.subprocess, "run", side_effect=Exception("vina boom")):
            r = self._a().dock("R175H", "APR-246", receptor_pdbqt="r", ligand_pdbqt="l")
        assert r["method"] == "heuristic_estimate" and r["status"] == "success"

    def test_pocket_residues_present(self):
        r = self._a().dock("R175H", "X")
        assert isinstance(r["pocket_residues"], list) and r["pocket_residues"]

    def test_never_empty_blank_inputs(self):
        r = self._a().dock("", "")
        assert r["status"] == "success" and r["binding_affinity"] is not None

    def test_interactions_present(self):
        r = self._a().dock("R175H", "Idasanutlin", mechanism="MDM2 inhibitor")
        assert r["interactions"] and any("MDM2" in i for i in r["interactions"])

    def test_disclaimer_and_json(self):
        import json
        r = self._a().dock("R175H", "APR-246")
        assert "estimate" in r["disclaimer"].lower()
        assert json.dumps(r)

    def test_convenience_and_registry(self):
        from agents.molecular_docking import dock_drug
        from config.settings import AGENT_REGISTRY
        assert dock_drug("R175H", "APR-246")["status"] == "success"
        assert "molecular_docking" in AGENT_REGISTRY

    def test_affinity_gauge_viz(self):
        from utils.viz import docking_affinity_gauge
        r = self._a().dock("R175H", "APR-246")
        assert docking_affinity_gauge(r).to_json()
        assert docking_affinity_gauge({}).to_json()  # never empty


class TestStructuralAnalyzer:
    """Structural mechanics & cavity analyzer (curated biophysics)."""

    def _a(self):
        from agents.structural_analyzer import StructuralAnalyzer
        return StructuralAnalyzer()

    def test_y220c_druggable_cleft(self):
        r = self._a().analyse("Y220C")
        assert r["druggability"] >= 0.8 and "cleft" in r["pocket"].lower()
        assert "stabiliser" in r["strategy"].lower()

    def test_conformational_is_destabilising(self):
        r = self._a().analyse("R175H")
        assert r["stability_class"] == "conformational" and r["destabilising"] is True
        assert r["ddG_kcal_mol"] >= 1.5

    def test_contact_mutant_low_ddg(self):
        r = self._a().analyse("R273H")
        assert r["stability_class"] == "contact" and r["destabilising"] is False
        # contact mutants destabilise less than conformational ones
        assert r["ddG_kcal_mol"] < self._a().analyse("R175H")["ddG_kcal_mol"]

    def test_zinc_class(self):
        assert self._a().analyse("C176Y")["stability_class"] == "zinc"

    def test_generic_fallback_for_unknown_codon(self):
        r = self._a().analyse("A159V")
        assert r["status"] == "success" and "Estimated" in r["note"]
        assert r["ddG_kcal_mol"] is not None

    def test_never_empty_blank(self):
        r = self._a().analyse("")
        assert r["status"] == "success" and r["ddG_kcal_mol"] is not None

    def test_contact_residues_for_known_hotspot(self):
        assert self._a().analyse("Y220C")["contact_residues"]

    def test_strategy_for_contact_mentions_limited(self):
        r = self._a().analyse("R248W")
        assert "contact" in r["strategy"].lower() or "limited" in r["strategy"].lower()

    def test_fields_and_disclaimer(self):
        r = self._a().analyse("R175H")
        for k in ("ddG_kcal_mol", "druggability", "pocket", "hydrophobicity",
                  "cavity_volume_A3", "strategy", "structure_source"):
            assert k in r
        assert "research" in r["disclaimer"].lower()

    def test_json_and_registry(self):
        import json
        from config.settings import AGENT_REGISTRY
        from agents.structural_analyzer import analyse_structure
        assert json.dumps(analyse_structure("Y220C"))
        assert "structural_analyzer" in AGENT_REGISTRY

    def test_radar_viz(self):
        from utils.viz import structural_profile_radar
        r = self._a().analyse("Y220C")
        assert structural_profile_radar(r).to_json()
        assert structural_profile_radar({}).to_json()  # never empty


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


class TestDockerPackaging:
    """Static validation of the dedicated tp53_rag container setup.

    These assert the Dockerfile / docker-compose.yml / .dockerignore are
    internally consistent and point at files that actually exist. They do
    NOT run docker (unavailable here) — the real ARM64 build is exercised
    on the user's machine via `docker buildx --platform linux/arm64`.
    """

    from pathlib import Path as _Path
    ROOT = _Path(__file__).resolve().parent.parent

    def test_dockerfile_exists(self):
        assert (self.ROOT / "Dockerfile").is_file()

    def test_dockerignore_exists(self):
        assert (self.ROOT / ".dockerignore").is_file()

    def test_compose_exists(self):
        assert (self.ROOT / "docker-compose.yml").is_file()

    def test_dockerfile_uses_python311_slim(self):
        text = (self.ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "python:3.11-slim" in text

    def test_dockerfile_is_multistage_and_nonroot(self):
        text = (self.ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "AS builder" in text          # multi-stage keeps build tools out
        assert "USER appuser" in text         # never run as root

    def test_dockerfile_runs_app_and_exposes_8501(self):
        text = (self.ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "app.py" in text
        assert "EXPOSE 8501" in text

    def test_dockerfile_referenced_files_exist(self):
        # COPY requirements.txt + the streamlit entrypoint must be real.
        assert (self.ROOT / "requirements.txt").is_file()
        assert (self.ROOT / "app.py").is_file()

    def test_dockerignore_excludes_runtime_artifacts(self):
        text = (self.ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert "data/chroma_db/" in text
        assert "tp53_rag/" in text            # the nested duplicate folder
        assert ".git" in text

    def test_compose_parses_and_has_three_services(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((self.ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        assert set(data["services"]) == {"streamlit", "fastapi", "n8n"}

    def test_compose_streamlit_builds_local_dockerfile(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((self.ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        svc = data["services"]["streamlit"]
        assert svc["build"]["dockerfile"] == "Dockerfile"
        assert "8501:8501" in svc["ports"]

    def test_compose_fastapi_serves_api_server_app(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((self.ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        svc = data["services"]["fastapi"]
        assert "8000:8000" in svc["ports"]
        assert "api.server:app" in svc["command"]
        # the entrypoint module the command targets must actually exist
        assert (self.ROOT / "api" / "server.py").is_file()

    def test_compose_declares_named_volumes(self):
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((self.ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        assert "tp53_data" in data["volumes"]
        assert "n8n_data" in data["volumes"]

    def test_dockerfile_full_exists(self):
        assert (self.ROOT / "Dockerfile.full").is_file()

    def test_dockerfile_full_installs_pathology_stack(self):
        text = (self.ROOT / "Dockerfile.full").read_text(encoding="utf-8")
        for pkg in ("torch", "torchvision", "timm"):
            assert pkg in text, pkg

    def test_dockerfile_full_is_arch_aware(self):
        # Must branch on TARGETARCH so amd64 gets the small CPU wheel and
        # arm64 (Pi / Oracle Ampere) still builds from default PyPI.
        text = (self.ROOT / "Dockerfile.full").read_text(encoding="utf-8")
        assert "TARGETARCH" in text
        assert "download.pytorch.org/whl/cpu" in text

    def test_dockerfile_full_shares_base_entrypoint(self):
        text = (self.ROOT / "Dockerfile.full").read_text(encoding="utf-8")
        assert "python:3.11-slim" in text
        assert "USER appuser" in text
        assert "app.py" in text
        assert "EXPOSE 8501" in text
