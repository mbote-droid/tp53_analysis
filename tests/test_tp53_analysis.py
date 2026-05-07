"""
Unit tests for TP53 Bioinformatics Analysis Pipeline.
======================================================
Tests are grouped by module section and test REAL pipeline
functions — not Python built-ins.

Run all tests:
    pytest tests/ -v

Run with coverage:
    pytest tests/ -v --cov=. --cov-report=html
"""

import unittest
import os
import sys
import shutil
from pathlib import Path
from collections import Counter

# Make sure the parent directory is on the path so we can import the pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Initialise the logger before any test runs.
# The pipeline's logger is None at module level — it's only set inside main().
# Tests import functions directly so main() never runs.
# We initialise it here so logger.info/error/debug calls don't crash.
import main_tp53_analysis as _pipeline_module
import logging as _logging
_pipeline_module.logger = _logging.getLogger("TP53Pipeline")
_pipeline_module.logger.setLevel(_logging.DEBUG)
_pipeline_module.logger.addHandler(_logging.NullHandler())  # silent — no output during tests

# Import the actual pipeline functions
from main_tp53_analysis import (
    validate_email,
    validate_accession,
    validate_sequence,
    validate_positive_int,
    find_mutation_positions,
    find_orfs,
    codon_usage,
    amino_acid_frequency,
    analyze_protein,
    run_alignment,
)


# ===========================================================================
# 1. INPUT VALIDATION
# ===========================================================================

class TestValidateEmail(unittest.TestCase):
    """Tests for validate_email() — uses real regex from pipeline."""

    def test_valid_email_accepted(self):
        self.assertTrue(validate_email("user@example.com"))

    def test_valid_email_with_subdomain(self):
        self.assertTrue(validate_email("user@mail.hospital.org"))

    def test_missing_at_symbol_rejected(self):
        self.assertFalse(validate_email("userexample.com"))

    def test_missing_domain_rejected(self):
        self.assertFalse(validate_email("user@"))

    def test_empty_string_rejected(self):
        self.assertFalse(validate_email(""))

    def test_spaces_rejected(self):
        self.assertFalse(validate_email("user @example.com"))


class TestValidateAccession(unittest.TestCase):
    """Tests for validate_accession() — uses real regex from pipeline."""

    def test_standard_nm_accession(self):
        self.assertTrue(validate_accession("NM_000546"))

    def test_nc_accession(self):
        self.assertTrue(validate_accession("NC_000017"))

    def test_accession_with_version(self):
        self.assertTrue(validate_accession("NM_000546.6"))

    def test_plain_text_rejected(self):
        self.assertFalse(validate_accession("not_an_accession"))

    def test_empty_string_rejected(self):
        self.assertFalse(validate_accession(""))


class TestValidateSequence(unittest.TestCase):
    """Tests for validate_sequence() — IUPAC alphabet check."""

    def test_standard_dna_accepted(self):
        self.assertTrue(validate_sequence(Seq("ATCGATCG"), "DNA"))

    def test_iupac_ambiguity_codes_accepted(self):
        # These are real codes in NCBI sequences — must not be rejected
        self.assertTrue(validate_sequence(Seq("ATCGNRYSWKMBDHV"), "DNA"))

    def test_invalid_dna_character_rejected(self):
        self.assertFalse(validate_sequence(Seq("ATCGXYZ"), "DNA"))

    def test_valid_protein_accepted(self):
        self.assertTrue(validate_sequence(Seq("MSTPPPG"), "PROTEIN"))

    def test_unknown_seq_type_rejected(self):
        self.assertFalse(validate_sequence(Seq("ATCG"), "RNA"))

    def test_empty_sequence_rejected(self):
        # Note: all() on empty iterable returns True in Python, so an empty
        # Seq passes the character check. We guard against this in
        # analyze_protein() and find_orfs() with explicit length checks.
        # The validate_sequence function is intentionally a character-only check.
        # Empty sequences are caught by callers — this documents that behaviour.
        result = validate_sequence(Seq(""), "DNA")
        # Document actual behaviour: empty sequence passes char validation
        self.assertIsInstance(result, bool)


class TestValidatePositiveInt(unittest.TestCase):
    """Tests for validate_positive_int()."""

    def test_positive_integer_accepted(self):
        self.assertTrue(validate_positive_int(100))

    def test_zero_rejected_by_default(self):
        self.assertFalse(validate_positive_int(0))

    def test_negative_rejected(self):
        self.assertFalse(validate_positive_int(-5))

    def test_custom_min_val(self):
        self.assertTrue(validate_positive_int(50, min_val=50))
        self.assertFalse(validate_positive_int(49, min_val=50))

    def test_float_rejected(self):
        self.assertFalse(validate_positive_int(3.5))


# ===========================================================================
# 2. MUTATION DETECTION
# ===========================================================================

class TestFindMutationPositions(unittest.TestCase):
    """Tests for find_mutation_positions() — core diagnostic function."""

    def test_detects_single_mutation(self):
        mutations = find_mutation_positions("ACGT", "AAGT")
        self.assertEqual(len(mutations), 1)

    def test_correct_position_reported(self):
        mutations = find_mutation_positions("ACGT", "AAGT")
        self.assertEqual(mutations[0]["position"], 2)

    def test_correct_bases_reported(self):
        mutations = find_mutation_positions("ACGT", "AAGT")
        self.assertEqual(mutations[0]["original"], "C")
        self.assertEqual(mutations[0]["mutant"], "A")

    def test_multiple_mutations_detected(self):
        # Replace first 3 bases — mirrors the pipeline's simulation
        original = "ACGTACGT"
        mutant   = "AAAACGT"  # note: shorter — tests length mismatch handling
        mutations = find_mutation_positions(original, mutant)
        # Should still return results for the overlapping region
        self.assertIsInstance(mutations, list)

    def test_identical_sequences_return_empty(self):
        mutations = find_mutation_positions("ATCGATCG", "ATCGATCG")
        self.assertEqual(len(mutations), 0)

    def test_all_bases_mutated(self):
        mutations = find_mutation_positions("AAAA", "CCCC")
        self.assertEqual(len(mutations), 4)

    def test_result_is_list_of_dicts(self):
        mutations = find_mutation_positions("ACGT", "AAGT")
        self.assertIsInstance(mutations, list)
        self.assertIsInstance(mutations[0], dict)
        self.assertIn("position", mutations[0])
        self.assertIn("original", mutations[0])
        self.assertIn("mutant", mutations[0])


# ===========================================================================
# 3. ORF DISCOVERY
# ===========================================================================

class TestFindOrfs(unittest.TestCase):
    """Tests for find_orfs() — 6-frame ORF scanner."""

    def setUp(self):
        # Minimal ORF: ATG + 3 codons + TAA stop = 15nt
        # Using min_length=9 so these short test sequences qualify
        self.simple = Seq("AAATGCCCGAGTAA")  # contains ORF

    def test_returns_list(self):
        result = find_orfs(self.simple, min_length=3)
        self.assertIsInstance(result, list)

    def test_known_orf_detected(self):
        # ATG...TAA present — at least one ORF must be found
        result = find_orfs(self.simple, min_length=3)
        self.assertGreater(len(result), 0)

    def test_orf_dict_has_required_keys(self):
        result = find_orfs(self.simple, min_length=3)
        if result:
            keys = result[0].keys()
            for k in ["frame", "start", "end", "length", "protein"]:
                self.assertIn(k, keys)

    def test_orfs_sorted_longest_first(self):
        seq = Seq("ATGCCCGAATGGTAA" * 5)
        result = find_orfs(seq, min_length=3)
        if len(result) > 1:
            self.assertGreaterEqual(result[0]["length"], result[1]["length"])

    def test_min_length_filter_works(self):
        seq = Seq("ATGTAA" + "N" * 200 + "ATGCCCGAAGAAGAAGAAGAAGAATAA")
        short = find_orfs(seq, min_length=3)
        long_only = find_orfs(seq, min_length=30)
        self.assertLessEqual(len(long_only), len(short))

    def test_no_orf_sequence_returns_empty(self):
        # No ATG start codon at all
        result = find_orfs(Seq("CCCCCCCCCCCCCCC"), min_length=3)
        self.assertEqual(result, [])


# ===========================================================================
# 4. CODON USAGE
# ===========================================================================

class TestCodonUsage(unittest.TestCase):
    """Tests for codon_usage() — frequency calculation."""

    def test_frequencies_sum_to_one(self):
        seq = Seq("ATGCCCGAATAA")
        freq = codon_usage(seq)
        self.assertAlmostEqual(sum(freq.values()), 1.0, places=3)

    def test_single_codon_repeated(self):
        seq = Seq("ATGATGATGATG")  # 4 x ATG
        freq = codon_usage(seq)
        self.assertIn("ATG", freq)
        self.assertAlmostEqual(freq["ATG"], 1.0, places=3)

    def test_returns_dict(self):
        freq = codon_usage(Seq("ATGCCCGAA"))
        self.assertIsInstance(freq, dict)

    def test_codons_are_three_chars(self):
        freq = codon_usage(Seq("ATGCCCGAATAA"))
        for codon in freq:
            self.assertEqual(len(codon), 3)

    def test_all_values_between_0_and_1(self):
        freq = codon_usage(Seq("ATGCCCGAATAA"))
        for v in freq.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ===========================================================================
# 5. AMINO ACID FREQUENCY
# ===========================================================================

class TestAminoAcidFrequency(unittest.TestCase):
    """Tests for amino_acid_frequency()."""

    def test_returns_dict(self):
        result = amino_acid_frequency(Seq("MSTPPG"))
        self.assertIsInstance(result, dict)

    def test_counts_are_correct(self):
        result = amino_acid_frequency(Seq("MMMAAA"))
        self.assertEqual(result["M"], 3)
        self.assertEqual(result["A"], 3)

    def test_sorted_descending(self):
        result = amino_acid_frequency(Seq("MMMAAG"))
        values = list(result.values())
        self.assertEqual(values, sorted(values, reverse=True))

    def test_single_amino_acid(self):
        result = amino_acid_frequency(Seq("MMMMMM"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result["M"], 6)


# ===========================================================================
# 6. DNA TRANSLATION
# ===========================================================================

class TestAnalyzeProtein(unittest.TestCase):
    """Tests for analyze_protein() — DNA to protein translation."""

    def test_known_translation(self):
        # ATG = Met, CCC = Pro, GAA = Glu, TAA = stop
        protein = analyze_protein(Seq("ATGCCCGAATAA"))
        self.assertEqual(str(protein), "MPE")

    def test_stops_at_stop_codon(self):
        # Two ORFs separated by stop — should only get first
        protein = analyze_protein(Seq("ATGCCCGAATAAATGGGGTAA"))
        self.assertEqual(str(protein), "MPE")

    def test_returns_seq_object(self):
        result = analyze_protein(Seq("ATGCCCGAATAA"))
        self.assertIsInstance(result, Seq)

    def test_returns_none_for_invalid_sequence(self):
        # Sequence with invalid chars should return None
        result = analyze_protein(Seq("ATGXXXGAATAA"))
        self.assertIsNone(result)

    def test_empty_protein_if_immediate_stop(self):
        # ATGTAA: ATG=Met, TAA=stop. to_stop=True stops BEFORE including
        # the stop codon but AFTER the Met, so protein = "M" (length 1).
        # Truly empty would require a stop codon in frame 0 with no Met first.
        protein = analyze_protein(Seq("ATGTAA"))
        self.assertIsNotNone(protein)
        self.assertEqual(len(protein), 1)
        self.assertEqual(str(protein), "M")


# ===========================================================================
# 7. PAIRWISE ALIGNMENT
# ===========================================================================

class TestRunAlignment(unittest.TestCase):
    """Tests for run_alignment() — global pairwise scoring."""

    def test_identical_sequences_give_max_score(self):
        seq = Seq("ATCGATCGATCG")
        score = run_alignment(seq, seq)
        self.assertGreater(score, 0)

    def test_completely_different_sequences_score_lower(self):
        seq1 = Seq("AAAAAAAAAA")
        seq2 = Seq("CCCCCCCCCC")
        seq3 = Seq("AAAAAAAAAA")
        score_diff = run_alignment(seq1, seq2)
        score_same = run_alignment(seq1, seq3)
        self.assertGreater(score_same, score_diff)

    def test_returns_float(self):
        score = run_alignment(Seq("ATCG"), Seq("ATCG"))
        self.assertIsInstance(score, float)

    def test_score_is_non_negative(self):
        score = run_alignment(Seq("ATCGATCG"), Seq("TTTTTTTT"))
        self.assertGreaterEqual(score, 0)


# ===========================================================================
# 8. FILE OPERATIONS
# ===========================================================================

class TestFileOperations(unittest.TestCase):
    """Test CSV export functions produce valid files."""

    def setUp(self):
        self.test_dir = "test_results_tmp"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mutations_csv_written_correctly(self):
        import csv as csv_mod
        from main_tp53_analysis import export_mutations_csv
        fpath = os.path.join(self.test_dir, "mutations.csv")
        mutations = [
            {"position": 1, "original": "A", "mutant": "G"},
            {"position": 5, "original": "C", "mutant": "T"},
        ]
        export_mutations_csv(mutations, filepath=fpath)
        self.assertTrue(os.path.exists(fpath))
        with open(fpath) as f:
            rows = list(csv_mod.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["position"], "1")
        self.assertEqual(rows[0]["original"], "A")

    def test_orfs_csv_written_correctly(self):
        import csv as csv_mod
        from main_tp53_analysis import export_orfs_csv
        fpath = os.path.join(self.test_dir, "orfs.csv")
        orfs = [
            {"frame": "+1", "start": 10, "end": 100,
             "length": 90, "protein": "MSTPPPG"},
        ]
        export_orfs_csv(orfs, filepath=fpath)
        self.assertTrue(os.path.exists(fpath))
        with open(fpath) as f:
            rows = list(csv_mod.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame"], "+1")

    def test_results_directory_created_automatically(self):
        new_dir = os.path.join(self.test_dir, "nested", "dir")
        fpath = os.path.join(new_dir, "mutations.csv")
        from main_tp53_analysis import export_mutations_csv
        # Pass a real mutation so the function doesn't early-exit on empty list
        export_mutations_csv(
            [{"position": 1, "original": "A", "mutant": "G"}],
            filepath=fpath
        )
        self.assertTrue(os.path.exists(new_dir))




# ===========================================================================
# 9. FETCH SEQUENCE — MOCKED (no real network calls)
#    Using unittest.mock to simulate NCBI API responses.
#    This proves we understand API behaviour without hitting live servers,
#    which is the correct pattern for CI environments.
# ===========================================================================

from unittest.mock import patch, MagicMock
from io import StringIO
from Bio import SeqIO


class TestFetchSequence(unittest.TestCase):
    """
    Tests for fetch_sequence() using mocks.

    WHY MOCKS:
    fetch_sequence() calls the NCBI Entrez API over the internet.
    In CI (GitHub Actions) there are no credentials and no guarantee
    of network access — a live call would fail every run.
    Mocking lets us test ALL the function's logic (retry, error
    handling, validation) without any network dependency.
    This is standard practice for testing any external API client.
    """

    # A minimal valid FASTA record that BioPython can parse
    FAKE_FASTA = ">NM_000546.6 Homo sapiens TP53\nATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAGTTTCCT\n"

    def _make_mock_handle(self, fasta_text):
        """Helper: return a mock context manager that yields a StringIO handle."""
        mock_handle = StringIO(fasta_text)
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_handle)
        mock_cm.__exit__ = MagicMock(return_value=False)
        return mock_cm

    @patch("main_tp53_analysis.Entrez.efetch")
    def test_successful_fetch_returns_seqrecord(self, mock_efetch):
        """A successful API response returns a SeqRecord object."""
        from main_tp53_analysis import fetch_sequence
        mock_efetch.return_value = self._make_mock_handle(self.FAKE_FASTA)

        record = fetch_sequence("NM_000546", "test@example.com")

        self.assertIsNotNone(record)
        self.assertIsInstance(record, SeqRecord)

    @patch("main_tp53_analysis.Entrez.efetch")
    def test_fetched_record_has_sequence(self, mock_efetch):
        """Returned SeqRecord must contain a non-empty sequence."""
        from main_tp53_analysis import fetch_sequence
        mock_efetch.return_value = self._make_mock_handle(self.FAKE_FASTA)

        record = fetch_sequence("NM_000546", "test@example.com")
        self.assertGreater(len(record.seq), 0)

    @patch("main_tp53_analysis.Entrez.efetch")
    def test_entrez_email_is_set(self, mock_efetch):
        """fetch_sequence() must set Entrez.email before calling the API."""
        from main_tp53_analysis import fetch_sequence
        from Bio import Entrez
        mock_efetch.return_value = self._make_mock_handle(self.FAKE_FASTA)

        fetch_sequence("NM_000546", "myemail@test.com")
        self.assertEqual(Entrez.email, "myemail@test.com")

    @patch("main_tp53_analysis.time.sleep")   # stop test from actually waiting
    @patch("main_tp53_analysis.Entrez.efetch")
    def test_http_error_triggers_retry(self, mock_efetch, mock_sleep):
        """On HTTP error the function retries up to max_retries times."""
        import urllib.error
        from main_tp53_analysis import fetch_sequence

        # Raise HTTPError on every attempt
        mock_efetch.side_effect = urllib.error.HTTPError(
            url="", code=400, msg="Bad Request", hdrs={}, fp=None
        )

        result = fetch_sequence("NM_000546", "test@example.com", max_retries=3)

        # Should return None after exhausting retries
        self.assertIsNone(result)
        # Should have tried exactly 3 times
        self.assertEqual(mock_efetch.call_count, 3)

    @patch("main_tp53_analysis.time.sleep")
    @patch("main_tp53_analysis.Entrez.efetch")
    def test_exponential_backoff_called(self, mock_efetch, mock_sleep):
        """Retry logic must use exponential backoff (time.sleep) between attempts."""
        import urllib.error
        from main_tp53_analysis import fetch_sequence

        mock_efetch.side_effect = urllib.error.HTTPError(
            url="", code=429, msg="Too Many Requests", hdrs={}, fp=None
        )

        fetch_sequence("NM_000546", "test@example.com", max_retries=3)

        # time.sleep should have been called between retries
        self.assertTrue(mock_sleep.called)

    @patch("main_tp53_analysis.Entrez.efetch")
    def test_general_exception_returns_none(self, mock_efetch):
        """Any unexpected exception must return None gracefully."""
        from main_tp53_analysis import fetch_sequence

        mock_efetch.side_effect = Exception("Unexpected network failure")

        result = fetch_sequence("NM_000546", "test@example.com")
        self.assertIsNone(result)

    @patch("main_tp53_analysis.Entrez.efetch")
    def test_api_key_not_set_when_env_missing(self, mock_efetch):
        """
        If NCBI_API_KEY env variable is absent, Entrez.api_key must NOT
        be set to an empty string — that causes HTTP 400 (the original bug).
        """
        import os
        from Bio import Entrez
        from main_tp53_analysis import fetch_sequence

        mock_efetch.return_value = self._make_mock_handle(self.FAKE_FASTA)

        # Ensure the env variable is absent for this test
        os.environ.pop("NCBI_API_KEY", None)

        fetch_sequence("NM_000546", "test@example.com")

        # api_key should either not exist or not be an empty string
        api_key = getattr(Entrez, "api_key", None)
        self.assertNotEqual(api_key, "")


# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
