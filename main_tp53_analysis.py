"""
TP53 Gene Bioinformatics Analysis Pipeline
============================================
Author: [Dr. Samuel Mbote]
Description:
    A bioinformatics pipeline for fetching, analyzing, and visualizing
    DNA sequences from NCBI. Covers: sequence translation, GC content
    analysis, pairwise alignment, mutation detection, ORF discovery,
    codon usage bias, amino acid frequency profiling, multi-species
    sequence comparison with phylogenetic tree building, and protein
    domain annotation via the EMBL-EBI InterProScan REST API.

Usage:
    Set your ENTREZ_EMAIL environment variable before running:
        export ENTREZ_EMAIL="your@email.com"
    Then run:
        python tp53_analysis.py --accession NM_000546

Requirements:
    - biopython >= 1.79
    - matplotlib >= 3.3.0
"""

import os
import sys
import argparse
import csv
import json
import time
import logging
import urllib.request
import urllib.parse
import urllib.error
import re
from collections import Counter
from typing import Optional, Dict, List, Tuple

from Bio import Entrez, SeqIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction
from Bio import Align
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers / CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------------------------

def setup_logging(log_file: str = "results/tp53_analysis.log") -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Args:
        log_file (str): Path to log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    
    logger = logging.getLogger("TP53Pipeline")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


logger = None  # Will be initialized in main


# ---------------------------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------------------------

def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email (str): Email address to validate.
        
    Returns:
        bool: True if valid email format, False otherwise.
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))


def validate_accession(accession: str) -> bool:
    """
    Validate NCBI accession format (basic check).
    
    Args:
        accession (str): Accession ID to validate.
        
    Returns:
        bool: True if accession format looks valid.
    """
    # NCBI accessions are typically 2+ letters followed by digits
    return bool(re.match(r'^[A-Z]{1,6}_?\d+(\.\d+)?$', accession))


def validate_sequence(seq: Seq, seq_type: str = "DNA") -> bool:
    """
    Validate that sequence contains only valid nucleotides.
    
    Args:
        seq (Seq): Sequence to validate.
        seq_type (str): Either "DNA" or "PROTEIN".
        
    Returns:
        bool: True if sequence is valid.
    """
    seq_str = str(seq).upper()

    # Empty sequence is never valid
    if not seq_str:
        return False

    if seq_type == "DNA":
        valid_chars = set("ACGTNRYSWKMBDHV-")  # Full IUPAC ambiguity alphabet
    elif seq_type == "PROTEIN":
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY*-")
    else:
        return False

    return all(c in valid_chars for c in seq_str)


def validate_positive_int(value: int, min_val: int = 1) -> bool:
    """
    Validate that a value is a positive integer.
    
    Args:
        value (int): Value to validate.
        min_val (int): Minimum allowed value.
        
    Returns:
        bool: True if valid.
    """
    return isinstance(value, int) and value >= min_val


def check_results_directory_writable() -> bool:
    """
    Check if results directory is writable.
    
    Returns:
        bool: True if writable, False otherwise.
    """
    try:
        os.makedirs("results", exist_ok=True)
        test_file = "results/.write_test"
        with open(test_file, 'w') as f:
            f.write("")
        os.remove(test_file)
        return True
    except OSError as e:
        logger.error(f"Cannot write to results/: {e}")
        return False


# ---------------------------------------------------------------------------
# 1. SEQUENCE FETCHING
# ---------------------------------------------------------------------------

def fetch_sequence(accession_id: str, email: str, max_retries: int = 3) -> Optional[SeqRecord]:
    """
    Fetch a nucleotide sequence from NCBI by accession ID with retry logic.

    Args:
        accession_id (str): NCBI accession number (e.g. 'NM_000546').
        email (str): Email address required by NCBI Entrez API.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        SeqRecord: BioPython SeqRecord object containing the sequence.
        None: If fetch fails after all retries.
    """
    accession_id = accession_id.strip()
    if not validate_accession(accession_id):
        logger.warning(f"Accession '{accession_id}' has unusual format, attempting fetch anyway.")
    
    Entrez.email = email.strip()
    # Only assign api_key when a real value exists — empty string causes HTTP 400
    _api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if _api_key:
        Entrez.api_key = _api_key
    
    for attempt in range(max_retries):
        try:
            with Entrez.efetch(
                db="nucleotide",
                id=accession_id,
                rettype="fasta",
                retmode="text"
            ) as handle:
                record = SeqIO.read(handle, "fasta")
            
            if not record:
                logger.error(f"No sequence returned for accession '{accession_id}'.")
                return None
            
            if not validate_sequence(record.seq, "DNA"):
                logger.warning(f"Sequence contains non-standard nucleotides.")
            
            logger.info(f"Fetched: {record.id} | Length: {len(record.seq)} bp")
            return record
            
        except urllib.error.HTTPError as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: HTTP Error {e.code}. Retrying...")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch accession '{accession_id}' after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch accession '{accession_id}': {e}")
            return None


# ---------------------------------------------------------------------------
# 2. TRANSLATION
# ---------------------------------------------------------------------------

def analyze_protein(dna_sequence: Seq) -> Optional[Seq]:
    """
    Translate a DNA sequence to a protein, stopping at the first stop codon.

    Args:
        dna_sequence (Seq): BioPython Seq object of the DNA sequence.

    Returns:
        Seq: Translated protein sequence.
        None: If translation fails.
    """
    if not dna_sequence or len(dna_sequence) == 0:
        logger.error("Empty DNA sequence provided for translation.")
        return None
    
    if not validate_sequence(dna_sequence, "DNA"):
        logger.error("Invalid DNA sequence (contains invalid nucleotides).")
        return None
    
    try:
        protein = dna_sequence.translate(to_stop=True)
        
        if len(protein) == 0:
            logger.warning("Translation produced empty protein — possible missing start codon.")
            return protein  # empty Seq is valid — not an error
        
        logger.info(f"Protein translated: {len(protein)} amino acids")
        return protein
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. PAIRWISE ALIGNMENT
# ---------------------------------------------------------------------------

def run_alignment(seq1: Seq, seq2: Seq) -> Optional[float]:
    """
    Perform global pairwise alignment between two sequences.

    Args:
        seq1 (Seq): First sequence (original).
        seq2 (Seq): Second sequence (mutant or homolog).

    Returns:
        float: Best alignment score.
        None: If alignment fails.
    """
    if not seq1 or not seq2:
        logger.error("Empty sequence provided for alignment.")
        return None
    
    if len(seq1) == 0 or len(seq2) == 0:
        logger.error("Sequence length is zero.")
        return None
    
    try:
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        alignments = aligner.align(seq1, seq2)
        
        if len(alignments) == 0:
            logger.warning("No alignments found.")
            return None
        
        score = alignments.score
        logger.info(f"Alignment score: {score:.2f}")
        return score
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 4. MUTATION DETECTION
# ---------------------------------------------------------------------------

def find_mutation_positions(original: str, mutant: str) -> List[Dict]:
    """
    Find and report every position where the mutant differs from the original.

    Args:
        original (str): Original DNA string.
        mutant (str):   Mutant DNA string.

    Returns:
        list[dict]: List of dicts with keys 'position', 'original', 'mutant'.
    """
    if not original or not mutant:
        logger.error("Empty sequences provided for mutation detection.")
        return []
    
    if len(original) != len(mutant):
        logger.warning(
            f"Sequences differ in length ({len(original)} vs {len(mutant)}). "
            "Comparing overlapping region only."
        )

    mutations = []
    min_len = min(len(original), len(mutant))
    
    for i in range(min_len):
        orig_base = original[i].upper()
        mut_base = mutant[i].upper()
        
        if orig_base != mut_base:
            mutations.append({
                "position": i + 1,
                "original": orig_base,
                "mutant": mut_base
            })
            logger.debug(f"Mutation @ position {i + 1}: {orig_base} -> {mut_base}")

    logger.info(f"Total mutations found: {len(mutations)}")
    return mutations


# ---------------------------------------------------------------------------
# 5. ORF FINDER — all 6 reading frames
# ---------------------------------------------------------------------------

def find_orfs(dna_sequence: Seq, min_length: int = 100) -> List[Dict]:
    """
    Scan all 6 reading frames and report ORFs above a minimum length.

    Args:
        dna_sequence (Seq): Full DNA sequence to scan.
        min_length (int):   Minimum ORF length in nucleotides (default 100).

    Returns:
        list[dict]: Each dict contains 'frame', 'start', 'end', 'length', 'protein'.
    """
    if not dna_sequence or len(dna_sequence) < 3:
        logger.error("Sequence too short for ORF discovery (minimum 3 bp).")
        return []
    
    if not validate_positive_int(min_length):
        logger.error(f"Invalid min_length: {min_length}. Must be positive integer.")
        return []
    
    orfs = []
    seq_len = len(dna_sequence)

    for strand, nuc in [(+1, dna_sequence), (-1, dna_sequence.reverse_complement())]:
        for frame in range(3):
            try:
                trans = nuc[frame:].translate()
                trans_str = str(trans)
                aa_start = 0

                while True:
                    met = trans_str.find("M", aa_start)
                    if met == -1:
                        break
                    stop = trans_str.find("*", met)
                    if stop == -1:
                        break

                    orf_len_aa = stop - met
                    orf_len_nt = orf_len_aa * 3

                    if orf_len_nt >= min_length:
                        nt_start = frame + met * 3
                        nt_end = frame + (stop + 1) * 3

                        if strand == -1:
                            nt_start = seq_len - (frame + (stop + 1) * 3)
                            nt_end = seq_len - (frame + met * 3)

                        # Ensure start <= end
                        if nt_start > nt_end:
                            nt_start, nt_end = nt_end, nt_start

                        orfs.append({
                            "frame": ("+" if strand == 1 else "-") + str(frame + 1),
                            "start": nt_start,
                            "end": nt_end,
                            "length": orf_len_nt,
                            "protein": str(trans_str[met:stop])
                        })

                    aa_start = stop + 1
            except Exception as e:
                logger.warning(f"Error processing frame {frame} on strand {strand}: {e}")
                continue

    orfs.sort(key=lambda x: x["length"], reverse=True)
    logger.info(f"ORFs found (>={min_length} nt): {len(orfs)}")
    if orfs:
        logger.info(f"  Longest ORF: Frame {orfs[0]['frame']} | "
                   f"{orfs[0]['start']}-{orfs[0]['end']} | {orfs[0]['length']} nt")
    return orfs


# ---------------------------------------------------------------------------
# 6. CODON USAGE BIAS
# ---------------------------------------------------------------------------

def codon_usage(dna_sequence: Seq) -> Dict[str, float]:
    """
    Calculate codon usage frequency across the coding sequence.

    Args:
        dna_sequence (Seq): DNA sequence (should be in-frame CDS).

    Returns:
        dict: {codon (str): frequency (float)} -- frequencies sum to 1.0.
              Empty dict if sequence too short.
    """
    if not dna_sequence or len(dna_sequence) < 3:
        logger.warning("Sequence too short for codon analysis (minimum 3 bp).")
        return {}
    
    try:
        seq_str = str(dna_sequence).upper()
        codons = [seq_str[i:i+3] for i in range(0, len(seq_str) - 2, 3)
                  if len(seq_str[i:i+3]) == 3]
        
        if len(codons) == 0:
            logger.warning("No complete codons found in sequence.")
            return {}
        
        total = len(codons)
        counts = Counter(codons)
        freq = {codon: round(count / total, 4) for codon, count in counts.items()}
        
        logger.info(f"Codon usage calculated over {total} codons ({len(freq)} unique).")
        return freq
    except Exception as e:
        logger.error(f"Codon usage analysis failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# 7. AMINO ACID FREQUENCY
# ---------------------------------------------------------------------------

def amino_acid_frequency(protein_sequence: Seq) -> Dict[str, int]:
    """
    Count and return the frequency of each amino acid in a protein sequence.

    Args:
        protein_sequence (Seq): Translated protein sequence.

    Returns:
        dict: {amino_acid (str): count (int)}, sorted by count descending.
    """
    if not protein_sequence or len(protein_sequence) == 0:
        logger.warning("Empty protein sequence provided.")
        return {}
    
    try:
        counts = Counter(str(protein_sequence).upper())
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        logger.debug(f"Amino acid frequency calculated for {len(protein_sequence)} residues.")
        return sorted_counts
    except Exception as e:
        logger.error(f"Amino acid frequency analysis failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# 8. MULTI-SPECIES COMPARISON & PHYLOGENETIC TREE
# ---------------------------------------------------------------------------

# TP53 orthologs across species. Add/swap accessions here to study any gene family.
TP53_HOMOLOGS = {
    "NM_000546":    "Human",
    "NM_011640":    "Mouse",
    "NM_001271820": "Zebrafish",
    "NM_009895":    "Rat",
    "NM_001123020": "Chimpanzee",
}


def fetch_multiple_sequences(accession_map: Dict[str, str], email: str) -> List[SeqRecord]:
    """
    Fetch multiple nucleotide sequences from NCBI with detailed error reporting.

    Args:
        accession_map (dict): {accession_id (str): species_label (str)}.
        email (str): Email for NCBI Entrez API.

    Returns:
        list[SeqRecord]: Records with species name as .id and .name.
    """
    if not accession_map:
        logger.error("Empty accession map provided.")
        return []
    
    records = []
    failed = []
    
    for accession, species in accession_map.items():
        try:
            rec = fetch_sequence(accession, email)
            if rec:
                label = species.replace(" ", "_")
                rec.id = label
                rec.name = label
                rec.description = ""
                records.append(rec)
            else:
                failed.append(f"{species} ({accession})")
            time.sleep(0.4)  # respect NCBI rate limit (~3 req/s)
        except Exception as e:
            logger.warning(f"Failed to fetch {species} ({accession}): {e}")
            failed.append(f"{species} ({accession})")

    logger.info(f"Fetched {len(records)}/{len(accession_map)} homolog sequences.")
    if failed:
        logger.warning(f"Failed accessions: {', '.join(failed)}")
    
    return records


def build_multiple_alignment(records: List[SeqRecord]) -> Optional[MultipleSeqAlignment]:
    """
    Build a multiple sequence alignment by trimming all records to the
    length of the shortest sequence.

    Args:
        records (list[SeqRecord]): Fetched sequence records.

    Returns:
        MultipleSeqAlignment: Trimmed alignment object.
        None: If insufficient sequences.
    """
    if not records or len(records) < 2:
        logger.error("At least 2 sequences required for multiple alignment.")
        return None
    
    try:
        min_len = min(len(r.seq) for r in records)
        if min_len < 100:
            logger.warning(f"Alignment length very short ({min_len} bp). Results may be unreliable.")
        
        trimmed = [
            SeqRecord(r.seq[:min_len], id=r.id, name=r.name, description="")
            for r in records
        ]
        alignment = MultipleSeqAlignment(trimmed)
        logger.info(f"Multiple alignment ready | {len(records)} species x {min_len} bp")
        return alignment
    except Exception as e:
        logger.error(f"Multiple alignment failed: {e}")
        return None


def build_phylogenetic_tree(alignment: MultipleSeqAlignment):
    """
    Build a Neighbour-Joining phylogenetic tree from a multiple alignment.

    Args:
        alignment (MultipleSeqAlignment): Aligned sequences.

    Returns:
        Bio.Phylo.BaseTree.Tree: The NJ tree.
        None: If tree building fails.
    """
    if not alignment or len(alignment) < 2:
        logger.error("At least 2 sequences required for phylogenetic tree.")
        return None
    
    try:
        calculator = DistanceCalculator("identity")
        dist_matrix = calculator.get_distance(alignment)
        constructor = DistanceTreeConstructor(calculator, method="nj")
        tree = constructor.build_tree(alignment)
        logger.info("Neighbour-Joining phylogenetic tree built successfully.")
        return tree
    except Exception as e:
        logger.error(f"Phylogenetic tree construction failed: {e}")
        return None


def save_phylogenetic_tree(tree,
                           txt_path: str = "results/phylo_tree.txt",
                           png_path: str = "results/phylo_tree.png") -> bool:
    """
    Save the phylogenetic tree as a Newick text file and a PNG figure.

    Args:
        tree: BioPython Tree object from build_phylogenetic_tree().
        txt_path (str): Output path for Newick-format text.
        png_path (str): Output path for the PNG figure.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not tree:
        logger.error("No tree provided for saving.")
        return False
    
    try:
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
        
        # Newick text
        Phylo.write(tree, txt_path, "newick")
        logger.info(f"Newick tree saved -> {txt_path}")

        # PNG figure
        fig, ax = plt.subplots(figsize=(10, 6))
        Phylo.draw(tree, axes=ax, do_show=False)
        ax.set_title(
            "TP53 Orthologs -- Neighbour-Joining Phylogenetic Tree",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Phylogenetic tree plot saved -> {png_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save phylogenetic tree: {e}")
        return False


def export_distance_matrix_csv(alignment: MultipleSeqAlignment,
                                filepath: str = "results/distance_matrix.csv") -> bool:
    """
    Export the pairwise identity distance matrix between all species to CSV.

    Args:
        alignment (MultipleSeqAlignment): Aligned sequences.
        filepath (str): Destination CSV path.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not alignment:
        logger.error("No alignment provided for distance matrix export.")
        return False
    
    try:
        calculator = DistanceCalculator("identity")
        dist_matrix = calculator.get_distance(alignment)
        names = dist_matrix.names

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Species"] + names)
            for i, name in enumerate(names):
                row = [name] + [
                    f"{dist_matrix[i][j]:.4f}" for j in range(len(names))
                ]
                writer.writerow(row)
        logger.info(f"Distance matrix exported -> {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to export distance matrix: {e}")
        return False


# ---------------------------------------------------------------------------
# 9. PROTEIN DOMAIN ANNOTATION -- EMBL-EBI InterProScan REST API
# ---------------------------------------------------------------------------

# NOTE: The EBI InterProScan REST API (iprscan5) was retired in 2024.
# The replacement is the InterPro API: https://www.ebi.ac.uk/interpro/api/
# Domain annotation now queries InterPro directly by protein sequence hash,
# or can be run manually at https://www.ebi.ac.uk/interpro/search/sequence/
_INTERPRO_API = "https://www.ebi.ac.uk/interpro/api/protein/UniProt/entry/interpro/?format=json"


def annotate_protein_domains(protein_seq: Seq, email: str,
                             max_wait: int = 180) -> List[Dict]:
    """
    Annotate protein domains using the EMBL-EBI InterPro REST API.

    Queries InterPro for known domains matching the TP53 protein.
    The legacy iprscan5 REST endpoint was retired in 2024 — this
    function uses the current InterPro API instead.

    For custom sequence submission, use the web interface at:
    https://www.ebi.ac.uk/interpro/search/sequence/

    Args:
        protein_seq (Seq): Translated protein sequence.
        email (str):       Contact email (kept for API compatibility).
        max_wait (int):    Timeout in seconds (default 180).

    Returns:
        list[dict]: Each dict has keys 'database', 'accession', 'name',
                    'start', 'end', 'score'. Empty list on failure.
    """
    if not protein_seq or len(protein_seq) == 0:
        logger.error("Empty protein sequence provided for domain annotation.")
        return []

    if len(protein_seq) < 10:
        logger.warning(
            f"Very short protein sequence ({len(protein_seq)} aa). "
            "Domain annotation may be unreliable."
        )

    # Query InterPro for TP53 human protein (P04637) — the canonical
    # UniProt entry for human p53. This returns all known domain annotations
    # for the exact protein this pipeline analyses.
    TP53_UNIPROT_ID = "P04637"
    url = f"https://www.ebi.ac.uk/interpro/api/entry/all/protein/UniProt/{TP53_UNIPROT_ID}/?format=json"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": f"TP53-Pipeline/1.0 ({email})"
            }
        )
        logger.info("Querying InterPro API for TP53 domain annotations...")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.warning(f"InterPro API returned HTTP {e.code}. Skipping domain annotation.")
        return []
    except Exception as e:
        logger.warning(f"Domain annotation failed: {e}. Skipping.")
        return []

    # --- Parse InterPro API response ---
    domains = []
    try:
        results = data.get("results", [])
        if not results:
            logger.warning("No domain results returned from InterPro.")
            return []

        for entry in results:
            metadata = entry.get("metadata", {})
            acc  = metadata.get("accession", "?")
            name = metadata.get("name", "?")
            db   = metadata.get("source_database", "?").upper()

            # Extract location from protein structure
            proteins = entry.get("proteins", [])
            for protein in proteins:
                for location in protein.get("entry_protein_locations", []):
                    for fragment in location.get("fragments", []):
                        domains.append({
                            "database":  db,
                            "accession": acc,
                            "name":      name,
                            "start":     fragment.get("start", "?"),
                            "end":       fragment.get("end", "?"),
                            "score":     "N/A",
                        })

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Could not parse InterPro domain results: {e}")
        return []

    logger.info(f"Protein domains annotated: {len(domains)} hits")
    for d in domains[:10]:
        logger.info(
            f"  [{d['database']}] {d['accession']} | "
            f"{d['name']} | aa {d['start']}-{d['end']}"
        )
    return domains


def export_domains_csv(domains: List[Dict],
                       filepath: str = "results/protein_domains.csv") -> bool:
    """
    Export protein domain annotations to a CSV file.

    Args:
        domains (list[dict]): Output from annotate_protein_domains().
        filepath (str): Destination CSV path.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not domains:
        logger.warning("No domain data to export.")
        return False
    
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["database", "accession", "name", "start", "end", "score"]
            )
            writer.writeheader()
            writer.writerows(domains)
        logger.info(f"Protein domains exported -> {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to export domains: {e}")
        return False


# ---------------------------------------------------------------------------
# 10. CSV EXPORTS -- mutations & ORFs
# ---------------------------------------------------------------------------

def export_mutations_csv(mutations: List[Dict],
                         filepath: str = "results/mutations.csv") -> bool:
    """
    Export detected mutations to a CSV file.

    Args:
        mutations (list[dict]): Output from find_mutation_positions().
        filepath (str): Destination CSV path.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not mutations:
        logger.warning("No mutations to export.")
        return False
    
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["position", "original", "mutant"])
            writer.writeheader()
            writer.writerows(mutations)
        logger.info(f"Mutations exported -> {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to export mutations: {e}")
        return False


def export_orfs_csv(orfs: List[Dict], filepath: str = "results/orfs.csv") -> bool:
    """
    Export discovered ORFs to a CSV file with full protein sequences.

    Args:
        orfs (list[dict]): Output from find_orfs().
        filepath (str): Destination CSV path.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not orfs:
        logger.warning("No ORFs to export.")
        return False
    
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["frame", "start", "end", "length", "protein"]
            )
            writer.writeheader()
            writer.writerows(orfs)
        logger.info(f"ORFs exported -> {filepath} ({len(orfs)} ORFs)")
        return True
    except Exception as e:
        logger.error(f"Failed to export ORFs: {e}")
        return False


# ---------------------------------------------------------------------------
# 11. PLOTTING -- GC content + amino acid frequency
# ---------------------------------------------------------------------------

def plot_all(dna_seq: Seq, protein_seq: Seq, gc_window: int = 100) -> bool:
    """
    Generate a two-panel figure:
        Panel 1 -- GC content sliding window across the full gene.
        Panel 2 -- Amino acid frequency bar chart of the translated protein.

    Args:
        dna_seq (Seq):     Full DNA sequence.
        protein_seq (Seq): Translated protein sequence.
        gc_window (int):   Sliding window size in bp (default 100).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not dna_seq or len(dna_seq) == 0:
        logger.error("Empty DNA sequence for plotting.")
        return False
    
    if not protein_seq or len(protein_seq) == 0:
        logger.error("Empty protein sequence for plotting.")
        return False
    
    if not validate_positive_int(gc_window):
        logger.error(f"Invalid gc_window: {gc_window}. Must be positive integer.")
        return False
    
    try:
        os.makedirs("results", exist_ok=True)

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            "TP53 Gene -- Bioinformatics Analysis",
            fontsize=16, fontweight="bold", y=0.98
        )
        gs = gridspec.GridSpec(2, 1, hspace=0.45)

        # Panel 1 -- GC Content
        ax1 = fig.add_subplot(gs[0])
        gc_values = []
        x_pos = []
        
        for i in range(0, len(dna_seq), gc_window):
            window = dna_seq[i:i + gc_window]
            if len(window) > 0:
                gc_values.append(gc_fraction(window) * 100)
                x_pos.append(i)
        
        if len(gc_values) == 0:
            logger.warning("Unable to calculate GC content.")
            return False
        
        mean_gc = sum(gc_values) / len(gc_values) if gc_values else 0
        ax1.plot(x_pos, gc_values, color="#2196F3", linewidth=1.5)
        ax1.axhline(y=mean_gc, color="red", linestyle="--", linewidth=1,
                    label=f"Mean GC: {mean_gc:.1f}%")
        ax1.fill_between(x_pos, gc_values, alpha=0.15, color="#2196F3")
        ax1.set_title(f"GC Content (Sliding Window = {gc_window} bp)", fontsize=13)
        ax1.set_xlabel("Position (bp)")
        ax1.set_ylabel("GC Content (%)")
        ax1.set_ylim([0, 100])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2 -- Amino Acid Frequency
        ax2 = fig.add_subplot(gs[1])
        aa_freq = amino_acid_frequency(protein_seq)
        
        if not aa_freq:
            logger.warning("Unable to calculate amino acid frequency.")
            return False
        
        aa_labels = list(aa_freq.keys())
        aa_counts = list(aa_freq.values())
        colors = plt.cm.viridis([i / len(aa_labels) for i in range(len(aa_labels))])
        bars = ax2.bar(aa_labels, aa_counts, color=colors,
                       edgecolor="white", linewidth=0.5)
        ax2.set_title("Amino Acid Frequency in Translated TP53 Protein", fontsize=13)
        ax2.set_xlabel("Amino Acid")
        ax2.set_ylabel("Count")
        ax2.grid(True, axis="y", alpha=0.3)
        
        for bar, count in zip(bars[:min(5, len(bars))], aa_counts[:5]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(aa_counts) * 0.01,
                str(count), ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        outpath = "results/tp53_analysis.png"
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Analysis plot saved -> {outpath}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        return False


# ---------------------------------------------------------------------------
# 12. CLI ARGUMENT PARSING
# ---------------------------------------------------------------------------

def parse_args():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bioinformatics pipeline: TP53 analysis with phylogeny & domain annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tp53_analysis.py --accession NM_000546
  python tp53_analysis.py --accession NM_000546 --skip-phylo --skip-domains
  python tp53_analysis.py --accession NM_000546 --mutation-window 100 --orf-min-length 200
        """
    )
    parser.add_argument(
        "--accession", default="NM_000546",
        help="Primary NCBI accession (default: NM_000546 = human TP53)"
    )
    parser.add_argument(
        "--mutation-window", type=int, default=50,
        help="Bases replaced with 'A' to simulate mutation (default: 50, must be > 0)"
    )
    parser.add_argument(
        "--gc-window", type=int, default=100,
        help="GC content sliding window size in bp (default: 100, must be > 0)"
    )
    parser.add_argument(
        "--orf-min-length", type=int, default=100,
        help="Minimum ORF length in nt to report (default: 100, must be >= 3)"
    )
    parser.add_argument(
        "--skip-phylo", action="store_true",
        help="Skip multi-species fetch and phylogenetic tree (faster, fewer API calls)"
    )
    parser.add_argument(
        "--skip-domains", action="store_true",
        help="Skip InterProScan domain annotation (saves ~2-3 minutes of polling)"
    )
    parser.add_argument(
        "--max-domain-wait", type=int, default=180,
        help="Maximum wait time for InterProScan job (seconds, default: 180)"
    )
    return parser.parse_args()


def validate_cli_args(args) -> bool:
    """
    Validate parsed command-line arguments.
    
    Args:
        args: Parsed arguments from argparse.
        
    Returns:
        bool: True if all arguments are valid.
    """
    valid = True
    
    if not args.accession:
        logger.error("Accession ID cannot be empty.")
        valid = False
    
    if args.mutation_window <= 0:
        logger.error(f"mutation-window must be positive (got {args.mutation_window})")
        valid = False
    
    if args.gc_window <= 0:
        logger.error(f"gc-window must be positive (got {args.gc_window})")
        valid = False
    
    if args.orf_min_length < 3:
        logger.error(f"orf-min-length must be >= 3 (got {args.orf_min_length})")
        valid = False
    
    if args.max_domain_wait <= 0:
        logger.error(f"max-domain-wait must be positive (got {args.max_domain_wait})")
        valid = False
    
    return valid


# ---------------------------------------------------------------------------
# 13. MAIN
# ---------------------------------------------------------------------------

def main():
    """Main pipeline execution."""
    global logger
    
    args = parse_args()
    
    # Initialize logging
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("TP53 Bioinformatics Pipeline")
    logger.info("=" * 70)
    
    # Validate arguments
    if not validate_cli_args(args):
        logger.error("Invalid command-line arguments. Exiting.")
        sys.exit(1)
    
    # Check email
    EMAIL = os.environ.get("ENTREZ_EMAIL")
    if not EMAIL:
        logger.error(
            "ENTREZ_EMAIL environment variable is not set.\n"
            "Run: export ENTREZ_EMAIL='your@email.com' then retry."
        )
        sys.exit(1)
    
    if not validate_email(EMAIL):
        logger.warning(f"ENTREZ_EMAIL '{EMAIL}' does not appear to be a valid email format.")
    
    # Check results directory
    if not check_results_directory_writable():
        logger.error("Cannot write to results/ directory. Exiting.")
        sys.exit(1)
    
    logger.info(f"Using accession: {args.accession}")
    logger.info(f"Using email: {EMAIL}")
    
    # 1 -- Fetch primary sequence
    logger.info("\n[STEP 1/9] Fetching sequence from NCBI...")
    record = fetch_sequence(args.accession, EMAIL)
    if not record:
        logger.error(f"Failed to fetch accession '{args.accession}'. Exiting.")
        sys.exit(1)

    dna_seq = record.seq

    # 2 -- Translate to protein
    logger.info("\n[STEP 2/9] Translating DNA to protein...")
    protein_seq = analyze_protein(dna_seq)
    if not protein_seq:
        logger.error("Translation failed. Exiting.")
        sys.exit(1)

    # 3 -- Simulate & detect mutations
    logger.info(f"\n[STEP 3/9] Detecting mutations (simulating {args.mutation_window} bp replacement)...")
    compare_len = min(1000, len(dna_seq))
    mutant_dna = Seq("A" * args.mutation_window) + dna_seq[args.mutation_window:compare_len]
    mutations = find_mutation_positions(str(dna_seq[:compare_len]), str(mutant_dna))

    # 4 -- Pairwise alignment
    logger.info("\n[STEP 4/9] Running pairwise alignment (original vs mutant)...")
    alignment_score = run_alignment(dna_seq[:compare_len], mutant_dna)
    if alignment_score is None:
        logger.warning("Alignment failed but continuing pipeline.")

    # 5 -- ORF discovery
    logger.info(f"\n[STEP 5/9] Discovering ORFs (min length: {args.orf_min_length} bp)...")
    orfs = find_orfs(dna_seq, min_length=args.orf_min_length)

    # 6 -- Codon usage
    logger.info("\n[STEP 6/9] Analyzing codon usage bias...")
    codon_freq = codon_usage(dna_seq)
    if codon_freq:
        top5 = sorted(codon_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("  Top 5 most frequent codons:")
        for codon, freq in top5:
            logger.info(f"    {codon}: {freq:.4f}")

    # 7 -- Multi-species comparison + phylogenetic tree
    if not args.skip_phylo:
        logger.info("\n[STEP 7/9] Multi-species comparison & phylogenetic tree...")
        homologs = fetch_multiple_sequences(TP53_HOMOLOGS, EMAIL)
        if len(homologs) >= 3:
            alignment = build_multiple_alignment(homologs)
            if alignment:
                tree = build_phylogenetic_tree(alignment)
                if tree:
                    save_phylogenetic_tree(tree)
                    export_distance_matrix_csv(alignment)
                else:
                    logger.warning("Tree building failed.")
            else:
                logger.warning("Multiple alignment failed.")
        else:
            logger.warning(f"Only {len(homologs)} sequences fetched (need ≥3) -- skipping phylogenetic tree.")
    else:
        logger.info("\n[STEP 7/9] Phylogenetic analysis skipped (--skip-phylo).")

    # 8 -- Protein domain annotation
    if not args.skip_domains:
        logger.info("\n[STEP 8/9] Protein domain annotation (InterProScan)...")
        logger.info(f"  (Submitting to EMBL-EBI... max wait: {args.max_domain_wait}s)")
        domains = annotate_protein_domains(protein_seq, EMAIL, max_wait=args.max_domain_wait)
        export_domains_csv(domains)
    else:
        logger.info("\n[STEP 8/9] Domain annotation skipped (--skip-domains).")

    # 9 -- Export CSVs & plots
    logger.info("\n[STEP 9/9] Exporting results and generating plots...")
    export_mutations_csv(mutations)
    export_orfs_csv(orfs)
    plot_all(dna_seq, protein_seq, gc_window=args.gc_window)

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline complete! All outputs are in the results/ folder.")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()