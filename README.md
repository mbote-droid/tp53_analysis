# TP53 Bioinformatics Analysis Pipeline

A comprehensive bioinformatics pipeline for analyzing the TP53 gene (tumor suppressor protein p53) and its orthologs across species. This pipeline fetches sequences from NCBI, performs sequence analysis, detects mutations, discovers open reading frames, conducts multi-species phylogenetic comparisons, and annotates protein domains.

## Features

- **Sequence Fetching**: Download nucleotide sequences from NCBI Entrez
- **DNA Translation**: Translate DNA sequences to protein
- **Mutation Detection**: Identify and report nucleotide mutations
- **Pairwise Alignment**: Global sequence alignment with scoring
- **ORF Discovery**: Find all open reading frames in 6 reading frames
- **Codon Usage Analysis**: Calculate codon frequency bias
- **Amino Acid Profiling**: Amino acid composition analysis
- **Multi-Species Comparison**: Phylogenetic tree construction (Neighbour-Joining)
- **Protein Domain Annotation**: EMBL-EBI InterProScan integration (Pfam, SMART, PRINTS, ProSite)
- **Comprehensive Visualization**: GC content analysis and amino acid frequency plots
- **Robust Error Handling**: Graceful failure modes with detailed logging
- **CSV Exports**: All results exported to machine-readable formats

## Requirements

- Python 3.10+
- Internet connection (for NCBI Entrez and EMBL-EBI APIs)
- NCBI account email (required by NCBI Entrez API)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tp53_analysis.git
cd tp53_analysis
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set NCBI Email Environment Variable

The NCBI Entrez API requires an email address. Set it before running:

```bash
   $env:ENTREZ_EMAIL="your.email@example.com"
```

**Optional:** For faster NCBI requests, obtain an API key from [NCBI](https://www.ncbi.nlm.nih.gov/account/settings/#api-key-management) and set:

```bash
$env:NCBI_API_KEY="your_api_key_here"
```

## Usage

### Basic Usage

```bash
python main_tp53_analysis.py --accession NM_000546
```

This fetches the human TP53 sequence (NM_000546) and runs the complete pipeline.

### Advanced Usage

```bash
# Skip phylogenetic analysis (faster)
python main_tp53_analysis.py --accession NM_000546 --skip-phylo

# Skip domain annotation (saves 2-3 minutes)
python main_tp53_analysis.py --accession NM_000546 --skip-domains

# Skip both optional analyses
python main_tp53_analysis.py --accession NM_000546 --skip-phylo --skip-domains

# Custom mutation window (100 bp instead of 50)
python main_tp53_analysis.py --accession NM_000546 --mutation-window 100

# Custom ORF minimum length (200 bp instead of 100)
python main_tp53_analysis.py --accession NM_000546 --orf-min-length 200

# Custom GC content window (200 bp instead of 100)
python main_tp53_analysis.py --accession NM_000546 --gc-window 200

# Increase domain annotation timeout (default 180s)
python main_tp53_analysis.py --accession NM_000546 --max-domain-wait 300
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--accession` | `NM_000546` | NCBI nucleotide accession ID |
| `--mutation-window` | `50` | Number of bases to simulate as mutation (bp) |
| `--gc-window` | `100` | GC content sliding window size (bp) |
| `--orf-min-length` | `100` | Minimum ORF length to report (bp) |
| `--skip-phylo` | `False` | Skip phylogenetic tree analysis |
| `--skip-domains` | `False` | Skip InterProScan domain annotation |
| `--max-domain-wait` | `180` | Max wait time for InterProScan (seconds) |

## Output Files

All results are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `tp53_analysis.png` | Main analysis figure (GC content + amino acid frequency) |
| `mutations.csv` | Detected mutations (position, original base, mutant base) |
| `orfs.csv` | Discovered ORFs (frame, start, end, length, protein sequence) |
| `protein_domains.csv` | Protein domain annotations (database, accession, name, coordinates, score) |
| `phylo_tree.txt` | Phylogenetic tree in Newick format (for FigTree/iTOL) |
| `phylo_tree.png` | Phylogenetic tree visualization |
| `distance_matrix.csv` | Pairwise sequence distances between species |
| `tp53_analysis.log` | Complete execution log with timestamps |

## Supported NCBI Accessions

The pipeline includes TP53 orthologs for multi-species comparison:

| Accession | Species |
|-----------|---------|
| `NM_000546` | Human (Homo sapiens) |
| `NM_011640` | Mouse (Mus musculus) |
| `NM_001271820` | Zebrafish (Danio rerio) |
| `NM_009895` | Rat (Rattus norvegicus) |
| `NM_001123020` | Chimpanzee (Pan troglodytes) |

You can edit `TP53_HOMOLOGS` in the code to study any gene family.

## Example Workflow

```bash
# 1. Set email
$env:ENTREZ_EMAIL="your@email.com"

# 2. Run complete analysis
python main_tp53_analysis.py --accession NM_000546

# 3. Check results
ls -lh results/
cat results/tp53_analysis.log

# 4. View outputs
# - Open results/tp53_analysis.png in image viewer
# - Open results/phylo_tree.png for phylogenetic tree
# - Open results/*.csv in spreadsheet application
# - Open results/phylo_tree.txt in FigTree or iTOL for interactive exploration
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Missing email**: Pipeline exits with clear instruction to set ENTREZ_EMAIL
- **Failed sequence fetch**: Retries up to 3 times with exponential backoff
- **Invalid inputs**: CLI arguments validated before execution
- **Network errors**: Gracefully handles API timeouts and failures
- **Optional analyses**: Phylogenetic and domain annotation failures don't stop pipeline
- **File system errors**: Checks write permissions before starting

All errors are logged to `results/tp53_analysis.log` for troubleshooting.

## Dependencies

See `requirements.txt` for complete list. Main packages:

- **biopython** - Sequence analysis and phylogenetics
- **matplotlib** - Data visualization
- **urllib3** - HTTP requests (EMBL-EBI API)

## Performance Notes

- **Sequence fetching**: 1-5 seconds per sequence
- **Translation & analysis**: < 1 second
- **Phylogenetic tree**: 5-10 seconds (depends on sequence length)
- **Domain annotation**: 1-5 minutes (depends on EMBL-EBI queue)

**Total runtime**: ~2-6 minutes for complete pipeline (including domain annotation)

Use `--skip-domains` to reduce runtime to ~30 seconds.

## Troubleshooting

### ENTREZ_EMAIL not set
```
[ERROR] ENTREZ_EMAIL environment variable is not set.
Run: $env:ENTREZ_EMAIL='your.email@example.com' then retry.
```
**Solution**: Set the environment variable as shown in Installation section.

### HTTP 429 (Too Many Requests)
```
[WARNING] HTTP Error 429 during fetch
```
**Solution**: NCBI rate limit exceeded. The pipeline retries automatically. Wait 1-2 minutes before running again, or obtain an NCBI API key.

### InterProScan timeout
```
[WARNING] Domain annotation timed out after 180s. Skipping.
```
**Solution**: EMBL-EBI is busy. Increase timeout: `--max-domain-wait 300` or skip domains: `--skip-domains`

### Invalid accession format
```
[WARNING] Accession 'INVALID123' has unusual format, attempting fetch anyway.
```
**Solution**: Check NCBI accession format. Valid formats: `NM_000546`, `NC_000017`, etc.

### Cannot write to results/
```
[ERROR] Cannot write to results/ directory. Exiting.
```
**Solution**: Check permissions in current directory. Run from a writable location.

## Logging

Detailed logs are saved to `results/tp53_analysis.log`. Log levels:
- **INFO**: Major pipeline steps and results
- **WARNING**: Recoverable errors and skipped optional analyses
- **ERROR**: Critical failures that stop the pipeline
- **DEBUG**: Detailed information (file only, not console)

View logs:
```bash
cat results/tp53_analysis.log
tail -f results/tp53_analysis.log  # Live monitoring
```

## Testing

This project uses a comprehensive suite of automated integration and unit tests to ensure the accuracy of the TP53 genomic analysis.

Mutation Accuracy: Validates that the pipeline correctly identifies single-nucleotide variants and maps them to the correct genomic coordinates.

Sequence Integrity: Ensures DNA-to-Protein translation logic handles biological sequences correctly.

Data Validation: Implements strict checks for NCBI accession formats (e.g., NM_ series) and user metadata to prevent "garbage-in, garbage-out" errors.

Bioinformatics Metrics: Verified calculations for GC-content and Open Reading Frame (ORF) discovery against known gold-standard TP53 sequences.

### Run All Tests

```bash
python -m pytest 
```

### Run Tests with Coverage Report

```bash
python -m pytest --cov=. --cov-report=html
```

This generates a coverage report in `htmlcov/index.html`.

### Run Specific Test Class

```bash
python -m pytest tests/test_tp53_analysis.py::TestSequenceAnalysis
```

### Run with Verbose Output

```bash
python -m pytest -v
```

### Test Coverage Goal

- Current coverage: ~80% (core analysis functions)
- All critical functions have unit tests
- Integration tests cover end-to-end pipeline

### Writing New Tests

Add new test cases to `tests/test_tp53_analysis.py`:

```python
def test_my_new_function(self):
    """Test description."""
    result = my_function(test_input)
    self.assertEqual(result, expected_output)
```

Then run: `python -m pytest tests/test_tp53_analysis.py::TestMyClass::test_my_new_function`

---

## Code Quality & Linting

We use **pylint** to maintain code quality standards.

### Run Pylint Locally

```bash
python -m pylint tp53_analysis.py
```

### Pylint Configuration

Code style rules are defined in `.pylintrc`. Current settings:
- Max line length: 100 characters
- Min Python version: 3.8
- Enforces PEP 8 conventions

### Interpreting Pylint Score

- **10.0**: Perfect code (unlikely, but great!)
- **8.0+**: Good quality code
- **7.0+**: Acceptable, minor improvements suggested
- **<7.0**: Code needs review and improvements

### Auto-format Code with Black

```bash
python -m black main_tp53_analysis.py
```

Black automatically formats your code to PEP 8 standards.

### CI/CD Pipeline

Both pytest and pylint run automatically on every push via GitHub Actions:
- `.github/workflows/tests.yml` - Runs unit tests
- `.github/workflows/pylint.yml` - Checks code quality

Check the "Actions" tab on your GitHub repo to see results!

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional analysis methods (SNP detection, conservation scoring)
- More visualization options
- Support for other genes/proteins
- Performance optimizations
- Additional database integrations

## Citation

If you use this pipeline in research, please cite:

```bibtex
@software{tp53_pipeline,
  title={TP53 Bioinformatics Analysis Pipeline},
  author={Samuel Mbote},
  year={2026},
  url={https://github.com/mbote-driod/tp53_analysis}
}
```

## References

- NCBI Entrez: https://www.ncbi.nlm.nih.gov/
- Biopython: https://biopython.org/
- EMBL-EBI InterProScan: https://www.ebi.ac.uk/interpro/
- Pfam: https://pfam.xfam.org/
- SMART: http://smart.embl.de/

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Authors

- Samuel Mbote

## Acknowledgments

- NCBI for sequence databases
- EMBL-EBI for InterProScan and domain annotation services
- BioPython community for excellent bioinformatics tools

---

**Last Updated**: 2026-04-26  
**Version**: 1.0.0