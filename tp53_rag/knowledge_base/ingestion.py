"""
============================================================
TP53 RAG Platform - Ingestion Pipeline (Optimized)
============================================================
"""
import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_core.documents import Document
# BUG FIX 1: Import had a line-break after the comma before NCBI_API_KEY,
# which is a SyntaxError in Python. Must be on one line or use backslash/parens.
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCUMENTS_DIR,
    ENTREZ_EMAIL,
    NCBI_API_KEY,
)
from utils.logger import log

CURATED_TP53_KNOWLEDGE = [
    {
        "content": """TP53 Gene Overview:
        TP53 (Tumor Protein P53) is located on chromosome 17p13.1 and encodes the p53 protein, a transcription
factor of 393 amino acids. It is the most frequently mutated gene in human cancer, with somatic mutations
detected in over 50% of all human tumors. p53 is often called the 'guardian of the genome' because it
coordinates cellular responses to genotoxic stress including DNA damage, hypoxia, and oncogene activation.
        Primary functions include:
        - Transcriptional activation of genes involved in cell cycle arrest (CDKN1A/p21)
        - Apoptosis induction (BAX, PUMA, NOXA)
        - DNA repair coordination (GADD45) RefSeq mRNA: NM_000546.6 | UniProt: P04637""",
        "metadata": {"source": "curated", "category": "gene_overview", "gene": "TP53", "priority": "critical"}
    },
    {
        "content": """TP53 Protein Domain Architecture:
        The p53 protein (393 aa) contains five major functional domains:
        1. N-terminal Transactivation Domain (TAD): residues 1-67
        2. Proline-Rich Region (PRR): residues 67-98
        3. DNA-Binding Domain (DBD): residues 94-292 (DBD contains zinc coordination site Cys176, His179,
Cys238, Cys242 and six hotspot codons: R175, G245, R248, R249, R273, R282)
        4. Tetramerization Domain (TET): residues 323-356
        5. C-terminal Regulatory Domain (CTD): residues 363-393""",
        "metadata": {"source": "curated", "category": "protein_domains", "gene": "TP53", "priority": "critical"}
    },
    {
        "content": """TP53 Hotspot Mutations in Cancer:
        R175H (c.524G>A): Conformational mutant. Loss of DNA binding. Gain-of-function (GOF) activity.
        R248W (c.742C>T): Contact mutant. lost DNA major groove contact. Common in Li-Fraumeni syndrome.
        R273H (c.818G>A): Contact mutant. Loss of DNA backbone phosphate contact.
        G245S (c.733G>A): Conformational mutant. Disrupts L3 loop conformation.
        R249S (c.747G>C): Conformational mutant. Enriched in hepatocellular carcinoma from aflatoxin B1
exposure.""",
        "metadata": {"source": "curated", "category": "mutations", "gene": "TP53", "priority": "critical"}
    }
]


class TP53DocumentIngester:
    """Ingests medical data safely using standard request pools and split maps."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        log.info("TP53DocumentIngester initialised")

    def load_curated_knowledge(self) -> List[Document]:
        documents = []
        for item in CURATED_TP53_KNOWLEDGE:
            doc = Document(
                page_content=item["content"].strip(),
                metadata={**item["metadata"], "ingestion_source": "curated_embedded", "offline_available": True}
            )
            documents.append(doc)
        return documents

    def load_ncbi_gene_summary(self) -> List[Document]:
        if not ENTREZ_EMAIL:
            return []
        try:
            from Bio import Entrez
            Entrez.email = ENTREZ_EMAIL
            if NCBI_API_KEY:
                Entrez.api_key = NCBI_API_KEY
            handle = Entrez.efetch(db="gene", id="7157", rettype="gene_table", retmode="text")
            content = handle.read()
            handle.close()
            return [Document(
                page_content=f"NCBI Gene Summary for TP53:\n{content}",
                metadata={"source": "ncbi_entrez", "category": "gene_summary", "gene": "TP53", "offline_available": False}
            )]
        except Exception as e:
            log.warning(f"NCBI fetch skipped: {e}")
            return []

    def load_uniprot_annotation(self) -> List[Document]:
        try:
            # Using session manager with finite timeouts for hackathon performance stability
            with requests.Session() as session:
                resp = session.get("https://uniprot.org", timeout=10)
                resp.raise_for_status()
                lines = [line for line in resp.text.split("\n") if line.startswith(("CC", "FT", "DE", "GN"))]
                content = "\n".join(lines[:150])  # Strict capping preserves token budget boundaries
                return [Document(
                    page_content=f"UniProt P04637 functional annotations:\n{content}",
                    metadata={"source": "uniprot", "category": "protein_domains", "gene": "TP53", "offline_available": False}
                )]
        except Exception as e:
            log.warning(f"UniProt fallback initiated: {e}")
            return []

    def load_user_documents(self) -> List[Document]:
        docs = []
        p = Path(DOCUMENTS_DIR)
        if not p.exists():
            return docs
        for file in p.glob("**/*"):
            try:
                if file.suffix == ".pdf":
                    docs.extend(PyPDFLoader(str(file)).load())
                elif file.suffix in [".txt", ".md"]:
                    docs.extend(TextLoader(str(file), encoding="utf-8").load())
            except Exception as e:
                log.error(f"Failed loading file {file.name}: {e}")
        return docs

    def ingest_all(
        self,
        include_ncbi: bool = True,
        include_uniprot: bool = True,
        include_user_docs: bool = True,
    ) -> List[Document]:
        """Gathers all active medical knowledge elements and splits them into compliant chunks."""
        raw_docs = self.load_curated_knowledge()
        if include_ncbi:
            raw_docs.extend(self.load_ncbi_gene_summary())
        if include_uniprot:
            raw_docs.extend(self.load_uniprot_annotation())
        if include_user_docs:
            raw_docs.extend(self.load_user_documents())

        split_chunks = self.text_splitter.split_documents(raw_docs)
        # BUG FIX 2: log.info line had a rogue leading newline/space that broke indentation
        log.info(f"Ingestion lifecycle complete. Transformed raw contexts into {len(split_chunks)} chunks.")
        return split_chunks
