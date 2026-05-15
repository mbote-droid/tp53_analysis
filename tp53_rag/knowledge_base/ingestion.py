"""
============================================================
TP53 RAG Platform - Knowledge Base Ingestion Pipeline
============================================================
Fetches, processes, and chunks TP53 domain knowledge from
multiple authoritative sources into ChromaDB.

Sources ingested:
  1. Static curated TP53 knowledge (embedded — works offline)
  2. NCBI Gene summaries via Entrez API
  3. UniProt P53 functional annotation
  4. User-provided PDF/text documents

This offline-first design is intentional: the platform works
in resource-limited settings (clinics, field hospitals) where
internet may be unavailable — aligning with the hackathon's
Global Resilience and Health & Sciences tracks.
============================================================
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCUMENTS_DIR,
    ENTREZ_EMAIL,
    NCBI_API_KEY,
)
from utils.logger import log


# ── Curated TP53 Knowledge Base ───────────────────────────
# Embedded directly so the system works 100% offline.
# This is the core RAG grounding layer for Gemma 4.
CURATED_TP53_KNOWLEDGE = [
    {
        "content": """
        TP53 Gene Overview:
        TP53 (Tumor Protein P53) is located on chromosome 17p13.1 and encodes the p53 protein,
        a transcription factor of 393 amino acids. It is the most frequently mutated gene in human
        cancer, with somatic mutations detected in over 50% of all human tumors. p53 is often called
        the 'guardian of the genome' because it coordinates cellular responses to genotoxic stress
        including DNA damage, hypoxia, and oncogene activation.

        Primary functions include:
        - Transcriptional activation of genes involved in cell cycle arrest (CDKN1A/p21)
        - Apoptosis induction (BAX, PUMA, NOXA)
        - DNA repair coordination (GADD45)
        - Senescence induction
        - Metabolic regulation
        - Ferroptosis regulation

        Gene ID: 7157
        RefSeq mRNA: NM_000546.6
        UniProt: P04637
        """,
        "metadata": {
            "source": "curated",
            "category": "gene_overview",
            "gene": "TP53",
            "priority": "critical",
        },
    },
    {
        "content": """
        TP53 Protein Domain Architecture:
        The p53 protein (393 aa) contains five major functional domains:

        1. N-terminal Transactivation Domain (TAD): residues 1-67
           - TAD1 (1-40): interacts with MDM2, TFIID, CBP/p300
           - TAD2 (40-67): required for full transcriptional activation
           - Key residues: Leu22, Trp23, Leu25, Phe19

        2. Proline-Rich Region (PRR): residues 67-98
           - PXXP motifs important for apoptosis signalling
           - Interacts with SH3 domain proteins

        3. DNA-Binding Domain (DBD): residues 94-292
           - Most frequently mutated region in cancer (~90% of mutations)
           - Contains zinc coordination site (Cys176, His179, Cys238, Cys242)
           - Six hotspot codons: R175, G245, R248, R249, R273, R282
           - Hotspot mutations classified as: contact (R248W, R273H) or
             structural/conformational (R175H, G245S, R249S, R282W)

        4. Tetramerization Domain (TET): residues 323-356
           - p53 functions as a homotetramer (dimer of dimers)
           - Mutations here disrupt tetramerization and dominant-negative effects

        5. C-terminal Regulatory Domain (CTD): residues 363-393
           - Lysine-rich, subject to extensive post-translational modification
           - Acetylation, ubiquitination, sumoylation, methylation, phosphorylation
           - Regulates DNA binding and protein stability
        """,
        "metadata": {
            "source": "curated",
            "category": "protein_domains",
            "gene": "TP53",
            "priority": "critical",
        },
    },
    {
        "content": """
        TP53 Hotspot Mutations in Cancer:

        R175H (c.524G>A): Most common TP53 mutation (~5% of all TP53 mutations).
        Conformational mutant. Loss of DNA binding. Gain-of-function (GOF) activity.
        Found in breast, colorectal, lung, and ovarian cancers. Associated with poor
        prognosis. Dominant-negative over wild-type p53.

        R248W (c.742C>T): Contact mutant. Direct contact with DNA major groove lost.
        High frequency in Li-Fraumeni syndrome. Associated with osteosarcoma and
        breast cancer. Strong dominant-negative effect.

        R248Q (c.743G>A): Contact mutant. Similar to R248W but retains some residual
        DNA binding. Associated with colorectal and lung cancer.

        R273H (c.818G>A): Contact mutant. Loss of DNA backbone phosphate contact.
        Frequently found in colorectal and lung cancer. GOF promotes invasion/metastasis.

        R273C (c.817C>T): Contact mutant. Similar mechanism to R273H.

        G245S (c.733G>A): Conformational mutant. Disrupts L3 loop conformation.
        Found in gastric and ovarian cancers.

        R249S (c.747G>C): Conformational mutant. Associated with hepatocellular
        carcinoma, especially aflatoxin B1-exposed patients. Classic environmental
        mutational signature.

        R282W (c.844C>T): Conformational mutant. Disrupts H2 helix structure.
        Found in breast and lung cancers.

        Clinical significance: Hotspot mutations have different therapeutic
        implications. APR-246 (eprenetapopt) targets R175H and R248 mutants,
        restoring wild-type conformation. PRIMA-1MET is under clinical investigation.
        """,
        "metadata": {
            "source": "curated",
            "category": "mutations",
            "gene": "TP53",
            "priority": "critical",
        },
    },
    {
        "content": """
        TP53 Pathway and Regulation:

        MDM2-p53 Feedback Loop:
        - MDM2 is the primary negative regulator of p53
        - MDM2 is itself transcriptionally activated by p53 (negative feedback)
        - MDM2 ubiquitinates p53, targeting it for proteasomal degradation
        - MDMX (MDM4) cooperates with MDM2 to inhibit p53
        - In normal cells, p53 half-life is ~20 minutes
        - DNA damage activates ATM/ATR kinases → phosphorylate p53 at Ser15, Ser20
        - Phosphorylation disrupts MDM2-p53 interaction → p53 stabilised
        - p53 accumulates, tetramerises, binds DNA response elements

        Key p53 Target Genes:
        Cell cycle arrest: CDKN1A (p21), GADD45A, 14-3-3σ
        Apoptosis: BAX, PUMA (BBC3), NOXA (PMAIP1), APAF1, FAS, DR5
        DNA repair: GADD45A, XPC, DDB2
        Antioxidant: SESN1, SESN2, GPX1, TIGAR
        Metabolism: TIGAR, SCO2, GLS2
        Ferroptosis: SLC7A11 repression, GLS2 activation
        Senescence: PML, PAI-1 (SERPINE1)

        Gain-of-Function (GOF) mutations:
        Some TP53 missense mutations (especially hotspots) acquire oncogenic
        GOF activities independent of wild-type p53 loss:
        - Promote invasion and metastasis (interaction with p63, p73)
        - Activate oncogenic transcription programs
        - Interact with chromatin remodelling complexes (SWI/SNF)
        - Inhibit p63/p73 tumour suppressor activity
        """,
        "metadata": {
            "source": "curated",
            "category": "pathway",
            "gene": "TP53",
            "priority": "high",
        },
    },
    {
        "content": """
        TP53 Cross-Species Conservation:

        p53 is highly conserved across vertebrates, underscoring its fundamental
        role in genome integrity. Conservation analysis is critical for interpreting
        the functional impact of mutations.

        Conservation summary:
        - Human (Homo sapiens) NM_000546: Reference sequence, 393 aa
        - Chimpanzee (Pan troglodytes) NM_001123020: 98.7% identity to human
          Key difference: residue 72 (Pro72Arg polymorphism in humans)
        - Mouse (Mus musculus) NM_011640: 77% identity
          DNA-binding domain highly conserved; N and C termini diverged
        - Rat (Rattus norvegicus) NM_009895: 76% identity
        - Zebrafish (Danio rerio) NM_001271820: 53% identity
          DBD most conserved region (~70%), tetramerization domain conserved

        Evolutionary significance:
        - Hotspot positions (R175, R248, R273, R282) are invariant across mammals
        - This cross-species conservation confirms their structural/functional criticality
        - Species-specific differences in proline-rich region affect apoptosis efficiency
        - Zebrafish p53 has additional N-terminal extension not present in mammals

        Phylogenetic clustering:
        Human → Chimpanzee → Mouse/Rat → Zebrafish
        This pattern reflects known vertebrate phylogeny, validating pipeline accuracy.
        """,
        "metadata": {
            "source": "curated",
            "category": "phylogenetics",
            "gene": "TP53",
            "priority": "high",
        },
    },
    {
        "content": """
        TP53 Clinical Syndromes and Cancer Associations:

        Germline TP53 Mutations — Li-Fraumeni Syndrome (LFS):
        - Rare autosomal dominant cancer predisposition syndrome
        - Classic LFS criteria: proband with sarcoma <45y, first-degree relative
          with cancer <45y or sarcoma at any age
        - Tumour spectrum: soft tissue sarcomas, osteosarcoma, breast cancer,
          brain tumours, adrenocortical carcinoma, leukaemia
        - Germline hotspots: R248W, R248Q, R175H, R273H (same as somatic hotspots)
        - Penetrance: >90% lifetime cancer risk
        - Management: Whole-body MRI surveillance, enhanced breast screening,
          avoidance of radiation therapy where possible

        Somatic TP53 Mutations by Cancer Type (IARC/TCGA data):
        - Ovarian cancer (serous): ~96% mutation frequency
        - Triple-negative breast cancer: ~80%
        - Small cell lung cancer: ~75%
        - Colorectal cancer: ~60%
        - Head and neck squamous cell: ~72%
        - Bladder cancer: ~50%
        - Glioblastoma: ~30%
        - Hepatocellular carcinoma: ~30% (enriched for R249S from aflatoxin)
        - Acute myeloid leukaemia: ~10% (but associated with complex karyotype)

        Prognostic implications:
        - Generally associated with poor prognosis
        - Some mutation types (R175H, R248W/Q) have particularly poor outcomes
        - TP53 mut status predicts resistance to certain chemotherapies
        - Emerging as biomarker for immunotherapy response (TMB correlation)
        """,
        "metadata": {
            "source": "curated",
            "category": "clinical",
            "gene": "TP53",
            "priority": "critical",
        },
    },
    {
        "content": """
        TP53 Mutation Classification and Functional Impact:

        Classification by mechanism:
        1. Missense mutations (most common, ~75%):
           - Contact mutants: directly lose DNA contact (R248W, R273H, R273C)
           - Structural/conformational: disrupt p53 fold (R175H, G245S, R249S)
           - Both classes can have GOF activity

        2. Nonsense mutations (~10%): premature stop codon, truncated/absent protein

        3. Frameshift mutations (~10%): insertion/deletion causing reading frame shift

        4. Splice site mutations (~5%): aberrant splicing, altered or absent protein

        5. In-frame deletions/insertions (<1%): maintain reading frame

        Functional classification (IARC):
        - Class 1: Loss of transactivation (most hotspot missense)
        - Class 2: Partial loss (some transactivation retained)
        - Class 3: Wild-type-like (rare missense, polymorphisms)
        - Class 4: GOF (dominant-negative + new oncogenic activity)

        Impact on p53 structure:
        - Zinc-coordinating residues (C176, H179, C238, C242): loss of zinc →
          complete unfolding of DBD
        - L2/L3 loop mutations: direct DNA contact or loop geometry disrupted
        - S2/S2' sheet mutations: β-sheet scaffold disrupted → global unfolding
        - H1 helix mutations: helix-turn-helix motif disrupted

        Codon 72 polymorphism (Pro72Arg, rs1042522):
        - Common germline SNP, ~50% of population heterozygous
        - Arg72 variant: more efficient MDM2-mediated degradation,
          stronger apoptosis induction
        - Pro72 variant: enhanced autophagy, may affect chemotherapy response
        - No definitive cancer risk difference established
        """,
        "metadata": {
            "source": "curated",
            "category": "mutation_classification",
            "gene": "TP53",
            "priority": "high",
        },
    },
    {
        "content": """
        TP53 Therapeutic Targeting Strategies:

        Reactivation of mutant p53 (restore WT function):
        - APR-246 / Eprenetapopt (PRIMA-1MET): electrophilic compound that
          covalently modifies Cys277 in mutant p53, restoring WT conformation.
          FDA Breakthrough Designation for TP53-mutant MDS. Phase 3 trials ongoing.
        - PC14586: first selective small molecule targeting Y220C p53 mutant.
          In Phase 1/2 trials (solid tumours with Y220C mutation).
        - COTI-2: restores WT conformation to multiple p53 mutants.
        - Arsenic trioxide: aggregation reversal of some p53 mutants.

        MDM2/MDMX inhibition (restore WT p53 in MDM2-amplified tumours):
        - Nutlins / RG7112 / Idasanutlin: MDM2-p53 PPI inhibitors
        - ALRN-6924: stapled peptide, dual MDM2/MDMX inhibitor
        - Indicated for WT p53 tumours with MDM2 overexpression (liposarcoma, AML)

        Gene therapy:
        - Gendicine (China-approved): adenoviral TP53 gene delivery
          First approved gene therapy product (2003, NMPA)
        - ONYX-015 (dl1520): oncolytic adenovirus selective for p53-null tumours

        Synthetic lethality:
        - TP53-mutant tumours rely on G2/M checkpoint (vs G1/S in WT p53)
        - WEE1 inhibitors (adavosertib), CHK1 inhibitors exploit this dependency
        - PARP inhibitors in TP53-BRCA co-mutant contexts

        Immunotherapy:
        - p53 neoantigens (R175H, Y220C, R248W) are HLA-restricted immunogenic
        - p53 peptide vaccines in clinical development
        - TCR-T cell therapy targeting p53 hotspot neoantigens (NCI trials)
        """,
        "metadata": {
            "source": "curated",
            "category": "therapeutics",
            "gene": "TP53",
            "priority": "high",
        },
    },
    {
        "content": """
        TP53 Codon Usage and GC Content Biology:

        GC content of TP53 gene:
        - Human TP53 CDS: ~53% GC content (moderate-high)
        - Exons 5-8 encode the DBD and are particularly GC-rich (~58%)
        - CpG dinucleotides are hotspots for spontaneous mutation (deamination
          of 5-methylcytosine → thymine at CpG sites)
        - CpG-mediated C→T transitions account for ~25% of TP53 somatic mutations
        - R248W (CGA→TGA) and R273H (CGT→CAT) are at CpG dinucleotides

        Codon usage bias:
        - TP53 uses synonymous codons non-randomly
        - Codon optimisation affects translation speed, co-translational folding
        - Rare codon clusters near domain boundaries may act as translational pauses
          allowing correct domain folding
        - Synonymous mutations can affect splicing (exonic splice enhancers/silencers)
          — not all synonymous TP53 variants are truly neutral

        Open Reading Frames:
        - TP53 encodes multiple isoforms from alternative promoters and splicing
        - Full-length p53α (393 aa): canonical, most abundant
        - Δ40p53 (Δ40, p47): lacks first 39 aa (alternative P2 promoter)
        - Δ133p53: lacks first 132 aa (internal P2 promoter in intron 4)
        - β/γ isoforms: C-terminal truncations from alternative splicing of intron 9
        - These isoforms have distinct functions and expression patterns
        - Δ133p53 is pro-survival; overexpressed in cancer and ageing
        """,
        "metadata": {
            "source": "curated",
            "category": "codon_usage_gc",
            "gene": "TP53",
            "priority": "medium",
        },
    },
]


class TP53DocumentIngester:
    """
    Ingests TP53 domain knowledge from multiple sources into ChromaDB.

    Architecture:
        Curated knowledge (offline) → text splitter → embeddings → ChromaDB
        NCBI/UniProt (online, optional) → text splitter → embeddings → ChromaDB
        User PDFs/docs (optional) → PDF loader → text splitter → embeddings → ChromaDB
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        log.info("TP53DocumentIngester initialised")

    def load_curated_knowledge(self) -> List[Document]:
        """Load the embedded curated TP53 knowledge base (offline-first)."""
        documents = []
        for item in CURATED_TP53_KNOWLEDGE:
            doc = Document(
                page_content=item["content"].strip(),
                metadata={
                    **item["metadata"],
                    "ingestion_source": "curated_embedded",
                    "offline_available": True,
                },
            )
            documents.append(doc)
        log.info(f"Loaded {len(documents)} curated TP53 knowledge documents")
        return documents

    def load_ncbi_gene_summary(self) -> List[Document]:
        """
        Fetch TP53 gene summary from NCBI Gene via Entrez API.
        Falls back gracefully if offline.
        """
        if not ENTREZ_EMAIL:
            log.warning("ENTREZ_EMAIL not set — skipping NCBI gene summary fetch")
            return []

        try:
            from Bio import Entrez
            Entrez.email = ENTREZ_EMAIL
            if NCBI_API_KEY:
                Entrez.api_key = NCBI_API_KEY

            log.info("Fetching TP53 gene summary from NCBI...")
            handle = Entrez.efetch(db="gene", id="7157", rettype="gene_table", retmode="text")
            content = handle.read()
            handle.close()

            doc = Document(
                page_content=f"NCBI Gene Summary for TP53 (Gene ID: 7157):\n{content}",
                metadata={
                    "source": "ncbi_entrez",
                    "category": "gene_summary",
                    "gene": "TP53",
                    "gene_id": "7157",
                    "offline_available": False,
                },
            )
            log.info("Successfully fetched NCBI TP53 gene summary")
            return [doc]

        except Exception as e:
            log.warning(f"NCBI fetch failed (pipeline continues): {e}")
            return []

    def load_uniprot_annotation(self) -> List[Document]:
        """
        Fetch p53 UniProt functional annotation.
        Falls back gracefully if offline.
        """
        try:
            url = "https://rest.uniprot.org/uniprotkb/P04637.txt"
            log.info("Fetching UniProt P04637 annotation...")
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            # Extract functional sections from UniProt flat file
            lines = resp.text.split("\n")
            functional_lines = []
            for line in lines:
                if line.startswith(("CC", "FT", "DE", "GN", "KW")):
                    functional_lines.append(line)

            content = "\n".join(functional_lines[:200])  # cap at 200 lines

            doc = Document(
                page_content=f"UniProt P04637 (Human TP53) Annotation:\n{content}",
                metadata={
                    "source": "uniprot",
                    "uniprot_id": "P04637",
                    "category": "protein_annotation",
                    "gene": "TP53",
                    "offline_available": False,
                },
            )
            log.info("Successfully fetched UniProt p53 annotation")
            return [doc]

        except Exception as e:
            log.warning(f"UniProt fetch failed (pipeline continues): {e}")
            return []

    def load_user_documents(self, directory: Optional[Path] = None) -> List[Document]:
        """
        Load user-provided PDFs and text files from the documents directory.
        Drop your TP53 papers here: data/documents/
        """
        doc_dir = directory or DOCUMENTS_DIR
        documents = []

        if not doc_dir.exists() or not any(doc_dir.iterdir()):
            log.info(f"No user documents found in {doc_dir} — skipping")
            return []

        # PDF files
        pdf_files = list(doc_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "source": "user_document",
                        "filename": pdf_path.name,
                        "category": "research_paper",
                        "offline_available": True,
                    })
                documents.extend(docs)
                log.info(f"Loaded PDF: {pdf_path.name} ({len(docs)} pages)")
            except Exception as e:
                log.warning(f"Failed to load {pdf_path.name}: {e}")

        # Text files
        txt_files = list(doc_dir.glob("*.txt"))
        for txt_path in txt_files:
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "source": "user_document",
                        "filename": txt_path.name,
                        "offline_available": True,
                    })
                documents.extend(docs)
                log.info(f"Loaded text file: {txt_path.name}")
            except Exception as e:
                log.warning(f"Failed to load {txt_path.name}: {e}")

        log.info(f"Loaded {len(documents)} user documents from {doc_dir}")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks for embedding."""
        chunks = self.text_splitter.split_documents(documents)
        log.info(f"Split {len(documents)} documents into {len(chunks)} chunks "
                 f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        return chunks

    def ingest_all(
        self,
        include_ncbi: bool = True,
        include_uniprot: bool = True,
        include_user_docs: bool = True,
    ) -> List[Document]:
        """
        Full ingestion pipeline: load all sources, chunk, return.

        Args:
            include_ncbi: Fetch from NCBI (requires internet + ENTREZ_EMAIL)
            include_uniprot: Fetch from UniProt (requires internet)
            include_user_docs: Load from data/documents/

        Returns:
            List of chunked Document objects ready for embedding
        """
        log.info("═" * 60)
        log.info("  TP53 RAG Knowledge Base Ingestion Pipeline")
        log.info("═" * 60)

        all_documents = []

        # 1. Curated knowledge (always — offline-first)
        all_documents.extend(self.load_curated_knowledge())

        # 2. NCBI (online, optional)
        if include_ncbi:
            all_documents.extend(self.load_ncbi_gene_summary())
            time.sleep(0.5)  # NCBI rate limiting

        # 3. UniProt (online, optional)
        if include_uniprot:
            all_documents.extend(self.load_uniprot_annotation())

        # 4. User-provided documents
        if include_user_docs:
            all_documents.extend(self.load_user_documents())

        log.info(f"Total documents loaded: {len(all_documents)}")

        # Chunk all documents
        chunks = self.chunk_documents(all_documents)

        log.info(f"Ingestion complete: {len(chunks)} chunks ready for embedding")
        return chunks
